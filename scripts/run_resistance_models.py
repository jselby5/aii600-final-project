from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    PolynomialFeatures,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


# ============================================================
# 0. SETTINGS
# ============================================================

# Paths are relative to the current working directory.
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "resistance_model_data.csv"

N_SPLITS = 5
RANDOM_STATE = 42
N_TREES = 300

# Reserved for a later feature-representation experiment.
TEST_LOG_RE = False

TARGET = "C_T"

# An underprediction is penalized twice as heavily as an
# equal-sized overprediction.
UNDERPRED_WEIGHT = 2.0

RANKING_METRIC = "HullBalanced_AsymMAE_C_T"

# Output formatting only:
# calculations, model fitting, and ranking retain full precision.
SIGNIFICANT_FIGURES = 4
CSV_FLOAT_FORMAT = f"%.{SIGNIFICANT_FIGURES}g"

if UNDERPRED_WEIGHT < 1.0:
    raise ValueError(
        "UNDERPRED_WEIGHT must be at least 1.0."
    )


# ============================================================
# 1. OUTPUT-FORMATTING HELPERS
# ============================================================

def format_float(value):
    """Format a floating-point value to four significant figures."""
    return f"{value:.{SIGNIFICANT_FIGURES}g}"


def save_csv(frame, filename):
    """
    Save model outputs under outputs/.

    Floating-point values use four significant figures.
    Integer identifiers and counts remain integers.
    """
    output_path = OUTPUT_DIR / filename

    frame.to_csv(
        output_path,
        index=False,
        float_format=CSV_FLOAT_FORMAT,
    )

    return output_path


def format_plot_axis(value, position):
    """Format numeric plot-axis labels."""
    return format_float(value)


# ============================================================
# 2. READ PREPARED DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

geometry_features = [
    "Beta",
    "Lp/Bpx",
    "LCG",
    "Ap/Nabla^(2/3)",
    "Ta",
    "Theta_0",
    "CAP",
]

# Dataset definitions:
#
# Fn:
#   Displacement/volume-based Froude number.
#
# Re:
#   Calculated during preparation using:
#       V = Fn * sqrt(g * Nabla^(1/3))
#       Re = V * Lp / Visc
#
# Trim:
#   Derived from running drafts using the original convention.
#
# C_T:
#   Calculated from measured resistance and running Swet.
#
# LCG and CAP:
#   Retain their existing percentage representation.
#
# No ITTC friction calculation or friction-subtracted target.

feature_sets = {
    "Speed": [
        "Fn",
    ],

    "Hull_NoRe": geometry_features + [
        "Fn",
        "Nabla_Lp3",
        "Trim",
    ],

    "Hull_Re": geometry_features + [
        "Fn",
        "Re",
        "Nabla_Lp3",
        "Trim",
    ],
}

metadata_columns = [
    "source_excel_row",
    "Configuration",
    "Model",
    "Year",
    "hull_group",
]

# Validate the standard prepared-data schema before using it.
base_features = list(dict.fromkeys(
    feature
    for features in feature_sets.values()
    for feature in features
))

required_columns = list(dict.fromkeys(
    metadata_columns + base_features + [TARGET]
))

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Prepared CSV is missing columns: {missing_columns}"
    )

if df.empty:
    raise ValueError(
        "Prepared CSV contains no rows."
    )

if df[metadata_columns].isna().any().any():
    raise ValueError(
        "Prepared CSV contains missing identifiers."
    )

# Ensure modeling columns contain usable numeric data.
for column in base_features + [TARGET]:
    df[column] = pd.to_numeric(
        df[column],
        errors="raise",
    )

numeric_values = df[
    base_features + [TARGET]
].to_numpy(dtype=float)

if not np.isfinite(numeric_values).all():
    raise ValueError(
        "Prepared CSV contains non-finite modeling values. "
        "Rerun the preparation script."
    )

if TEST_LOG_RE:
    if (df["Re"] <= 0).any():
        raise ValueError(
            "Re must be positive for log10(Re)."
        )

    df["log10_Re"] = np.log10(df["Re"])

    # Replace raw Re rather than include both representations.
    feature_sets["Hull_LogRe"] = geometry_features + [
        "Fn",
        "log10_Re",
        "Nabla_Lp3",
        "Trim",
    ]

n_groups = df["hull_group"].nunique()

if n_groups < N_SPLITS:
    raise ValueError(
        f"{N_SPLITS} folds requested, but only "
        f"{n_groups} groups are available."
    )

y = df[TARGET].to_numpy(dtype=float)
groups = df["hull_group"].to_numpy()

print(f"Prepared dataset: {INPUT_FILE.resolve()}")
print(f"Output directory: {OUTPUT_DIR.resolve()}")
print(f"Rows: {len(df)}")
print(f"Groups: {n_groups}")
print(f"Cross-validation folds: {N_SPLITS}")
print(
    "Underprediction weight:",
    format_float(UNDERPRED_WEIGHT),
)
print(f"Ranking metric: {RANKING_METRIC}")


# ============================================================
# 3. MODEL / PARAMETER VARIATIONS
# ============================================================

model_specs = []


def add_model(name, family, parameters, estimator):
    model_specs.append({
        "name": name,
        "family": family,
        "parameters": parameters,
        "estimator": estimator,
    })


# ------------------------------------------------------------
# 3A. Linear regression
# ------------------------------------------------------------

add_model(
    name="Linear",
    family="Linear",
    parameters={},
    estimator=Pipeline([
        ("scale", StandardScaler()),
        ("model", LinearRegression()),
    ]),
)


# ------------------------------------------------------------
# 3B. Ridge regression
# ------------------------------------------------------------

for alpha in [0.1, 1.0, 10.0, 100.0]:
    add_model(
        name=f"Ridge_a{alpha:g}",
        family="Ridge",
        parameters={
            "alpha": alpha,
        },
        estimator=Pipeline([
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]),
    )


# ------------------------------------------------------------
# 3C. Quadratic features + Ridge
# ------------------------------------------------------------

for alpha in [1.0, 10.0, 100.0]:
    add_model(
        name=f"Poly2_Ridge_a{alpha:g}",
        family="PolynomialRidge",
        parameters={
            "degree": 2,
            "alpha": alpha,
        },
        estimator=Pipeline([
            ("input_scale", StandardScaler()),
            (
                "poly",
                PolynomialFeatures(
                    degree=2,
                    include_bias=False,
                ),
            ),
            ("term_scale", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]),
    )


# ------------------------------------------------------------
# 3D. Extra Trees
# ------------------------------------------------------------

for min_leaf in [2, 5, 10]:
    parameters = {
        "n_estimators": N_TREES,
        "min_samples_leaf": min_leaf,
        "max_features": 1.0,
    }

    add_model(
        name=f"ExtraTrees_leaf{min_leaf}",
        family="ExtraTrees",
        parameters=parameters,
        estimator=ExtraTreesRegressor(
            **parameters,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    )


# ------------------------------------------------------------
# 3E. Histogram gradient boosting
# ------------------------------------------------------------

boost_min_leaf = max(
    5,
    min(20, len(df) // 50),
)

for max_leaf_nodes in [7, 15]:
    for l2 in [0.0, 1.0]:
        parameters = {
            "max_iter": 300,
            "learning_rate": 0.05,
            "max_leaf_nodes": max_leaf_nodes,
            "min_samples_leaf": boost_min_leaf,
            "l2_regularization": l2,

            # Avoid a separate row-random validation split.
            "early_stopping": False,
        }

        add_model(
            name=(
                f"HistGB_leaves{max_leaf_nodes}"
                f"_l2{l2:g}"
            ),
            family="HistGradientBoosting",
            parameters=parameters,
            estimator=HistGradientBoostingRegressor(
                **parameters,
                loss="squared_error",
                random_state=RANDOM_STATE,
            ),
        )


# ============================================================
# 4. IDENTICAL GROUPED SPLITS FOR ALL EXPERIMENTS
# ============================================================

gkf = GroupKFold(n_splits=N_SPLITS)

splits = list(
    gkf.split(
        df,
        y,
        groups=groups,
    )
)

fold_assignment = np.full(
    len(df),
    -1,
    dtype=int,
)

for fold, (_, test_idx) in enumerate(
    splits,
    start=1,
):
    fold_assignment[test_idx] = fold

assignments = df[metadata_columns].copy()
assignments["fold"] = fold_assignment

save_csv(
    assignments,
    "cv_assignments.csv",
)


# ============================================================
# 5. METRICS
# ============================================================

def calculate_metrics(y_true, y_pred, group_ids):
    """
    Calculate ordinary and asymmetric coefficient errors.

    Sign convention:
        error = predicted - measured

    Asymmetric loss:
        underprediction: UNDERPRED_WEIGHT * absolute error
        overprediction:  absolute error

    Group-balanced metrics give equal weight to each
    Configuration–Model–Year group.

    No clipping is applied to predictions.
    """
    y_true = np.asarray(
        y_true,
        dtype=float,
    )
    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )
    group_ids = np.asarray(group_ids)

    error = y_pred - y_true
    absolute_error = np.abs(error)

    shortfall = np.maximum(
        -error,
        0.0,
    )
    excess = np.maximum(
        error,
        0.0,
    )

    under = error < 0

    asymmetric_error = (
        UNDERPRED_WEIGHT * shortfall
        + excess
    )

    group_metrics = (
        pd.DataFrame({
            "group": group_ids,
            "absolute_error": absolute_error,
            "asymmetric_error": asymmetric_error,
        })
        .groupby("group")[
            ["absolute_error", "asymmetric_error"]
        ]
        .mean()
    )

    return {
        "MAE_C_T": mean_absolute_error(
            y_true,
            y_pred,
        ),

        "RMSE_C_T": np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        ),

        "HullBalanced_MAE_C_T": (
            group_metrics["absolute_error"].mean()
        ),

        "AsymMAE_C_T": (
            asymmetric_error.mean()
        ),

        "HullBalanced_AsymMAE_C_T": (
            group_metrics["asymmetric_error"].mean()
        ),

        "Bias_C_T": error.mean(),

        "Underpred_Freq_pct": (
            100.0 * under.mean()
        ),

        # Conditional mean: underpredicted observations only.
        "Underpred_Mag_C_T": (
            shortfall[under].mean()
            if under.any()
            else 0.0
        ),

        # Unconditional mean: zero for observations that
        # do not underpredict.
        "MeanShortfall_C_T": (
            shortfall.mean()
        ),

        "NegativePrediction_pct": (
            100.0 * np.mean(y_pred < 0)
        ),
    }


# ============================================================
# 6. RUN EXPERIMENTS
# ============================================================

summary_rows = []
fold_rows = []
oof_frames = []

n_experiments = (
    len(feature_sets) * len(model_specs)
)

experiment_number = 0

for feature_set_name, features in feature_sets.items():

    X = df[features]

    for spec in model_specs:
        experiment_number += 1

        experiment_name = (
            f"{feature_set_name}__{spec['name']}"
        )

        print(
            f"[{experiment_number}/{n_experiments}] "
            f"{experiment_name}"
        )

        oof_prediction = np.full(
            len(df),
            np.nan,
        )

        fold_maes = []
        fold_ranking_scores = []

        for fold, (train_idx, test_idx) in enumerate(
            splits,
            start=1,
        ):
            estimator = clone(
                spec["estimator"]
            )

            estimator.fit(
                X.iloc[train_idx],
                y[train_idx],
            )

            prediction = np.asarray(
                estimator.predict(
                    X.iloc[test_idx]
                ),
                dtype=float,
            ).ravel()

            if not np.isfinite(prediction).all():
                raise ValueError(
                    "Non-finite predictions from "
                    f"{experiment_name}, fold {fold}."
                )

            oof_prediction[test_idx] = prediction

            metrics = calculate_metrics(
                y[test_idx],
                prediction,
                groups[test_idx],
            )

            fold_maes.append(
                metrics["MAE_C_T"]
            )
            fold_ranking_scores.append(
                metrics[RANKING_METRIC]
            )

            fold_rows.append({
                "Experiment": experiment_name,
                "FeatureSet": feature_set_name,
                "Model": spec["name"],
                "Fold": fold,
                "N_train": len(train_idx),
                "N_test": len(test_idx),
                **metrics,
            })

        if not np.isfinite(oof_prediction).all():
            raise RuntimeError(
                "Incomplete out-of-fold predictions: "
                f"{experiment_name}"
            )

        # Main metrics use all out-of-fold predictions.
        # They are not simple averages of fold metrics.
        pooled_metrics = calculate_metrics(
            y,
            oof_prediction,
            groups,
        )

        summary_rows.append({
            "Experiment": experiment_name,
            "FeatureSet": feature_set_name,
            "Model": spec["name"],
            "Family": spec["family"],

            "Parameters": json.dumps(
                spec["parameters"],
                sort_keys=True,
            ),

            "Underprediction_weight": UNDERPRED_WEIGHT,

            "N_features": len(features),
            "N_rows": len(df),
            "N_groups": n_groups,

            **pooled_metrics,

            "Fold_MAE_mean": np.mean(
                fold_maes
            ),
            "Fold_MAE_std": np.std(
                fold_maes,
                ddof=1,
            ),

            "Fold_RankingScore_mean": np.mean(
                fold_ranking_scores
            ),
            "Fold_RankingScore_std": np.std(
                fold_ranking_scores,
                ddof=1,
            ),
        })

        prediction_frame = assignments.copy()

        prediction_frame["Experiment"] = experiment_name
        prediction_frame["Fn"] = df["Fn"].to_numpy()
        prediction_frame["Re"] = df["Re"].to_numpy()
        prediction_frame["C_T_true"] = y
        prediction_frame["C_T_pred"] = oof_prediction

        oof_frames.append(
            prediction_frame
        )


# ============================================================
# 7. RANK AND SAVE RESULTS
# ============================================================

# Sorting occurs at full precision, before output formatting.
summary_df = (
    pd.DataFrame(summary_rows)
    .sort_values(
        [RANKING_METRIC, "RMSE_C_T"],
        ascending=True,
    )
    .reset_index(drop=True)
)

summary_df.insert(
    0,
    "Rank",
    np.arange(
        1,
        len(summary_df) + 1,
    ),
)

fold_df = pd.DataFrame(
    fold_rows
)

oof_df = pd.concat(
    oof_frames,
    ignore_index=True,
)

save_csv(
    summary_df,
    "model_summary.csv",
)

save_csv(
    fold_df,
    "fold_results.csv",
)

save_csv(
    oof_df,
    "oof_predictions.csv",
)


# ============================================================
# 8. SAVE EXPERIMENT CONFIGURATION
# ============================================================

config = {
    "input_file": str(INPUT_FILE.resolve()),
    "output_directory": str(OUTPUT_DIR.resolve()),
    "n_splits": N_SPLITS,
    "random_state": RANDOM_STATE,
    "test_log_re": TEST_LOG_RE,
    "target": TARGET,
    "ranking_metric": RANKING_METRIC,
    "underprediction_weight": UNDERPRED_WEIGHT,
    "output_significant_figures": SIGNIFICANT_FIGURES,

    "group_definition": [
        "Configuration",
        "Model",
        "Year",
    ],

    "feature_sets": feature_sets,

    "models": [
        {
            "name": spec["name"],
            "family": spec["family"],
            "parameters": spec["parameters"],
        }
        for spec in model_specs
    ],
}

with open(
    OUTPUT_DIR / "experiment_config.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        config,
        file,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# 9. DISPLAY LEADING RESULTS
# ============================================================

display_columns = [
    "Rank",
    "Experiment",
    "HullBalanced_AsymMAE_C_T",
    "HullBalanced_MAE_C_T",
    "RMSE_C_T",
    "Bias_C_T",
    "Underpred_Freq_pct",
    "Underpred_Mag_C_T",
    "NegativePrediction_pct",
]

print("\nLeading experiments:")

print(
    summary_df[display_columns]
    .head(15)
    .to_string(
        index=False,
        float_format=format_float,
    )
)


# ============================================================
# 10. MODEL-COMPARISON FIGURE
# ============================================================

top_n = min(
    15,
    len(summary_df),
)

top_df = (
    summary_df
    .head(top_n)
    .iloc[::-1]
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(17, 8),
)

# ------------------------------------------------------------
# Left: best individual experiments
# ------------------------------------------------------------

axes[0].barh(
    top_df["Experiment"],
    top_df[RANKING_METRIC],
    color="steelblue",
)

axes[0].set_xlabel(
    "Group-balanced asymmetric MAE\n"
    f"(underprediction weight = "
    f"{format_float(UNDERPRED_WEIGHT)})"
)

axes[0].set_title(
    f"Top {top_n} experiments"
)

axes[0].tick_params(
    axis="y",
    labelsize=8,
)

axes[0].xaxis.set_major_formatter(
    FuncFormatter(format_plot_axis)
)

# ------------------------------------------------------------
# Right: best setting for each family / feature-set combination
# ------------------------------------------------------------

family_comparison = summary_df.pivot_table(
    index="Family",
    columns="FeatureSet",
    values=RANKING_METRIC,
    aggfunc="min",
)

family_comparison.plot.bar(
    ax=axes[1],
    width=0.8,
)

axes[1].set_ylabel(
    "Group-balanced asymmetric MAE"
)

axes[1].set_xlabel(
    "Model family"
)

axes[1].set_title(
    "Best tested setting per family and feature set"
)

axes[1].tick_params(
    axis="x",
    rotation=35,
)

axes[1].yaxis.set_major_formatter(
    FuncFormatter(format_plot_axis)
)

axes[1].legend(
    title="Feature set"
)

fig.tight_layout()

# Absolute path and unique filename retain the successful
# workaround for the earlier image-saving error.
timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S_%f"
)

figure_path = (
    OUTPUT_DIR.resolve()
    / f"model_comparison_{timestamp}.png"
)

try:
    fig.savefig(
        str(figure_path),
        format="png",
        dpi=150,
        bbox_inches="tight",
    )

    print(
        f"\nFigure saved: {figure_path}"
    )

except OSError as exc:
    print(
        f"\nFigure could not be saved: {exc}"
    )
    print(
        "CSV results have already been saved."
    )

finally:
    plt.close(fig)


# ============================================================
# 11. COMPLETION NOTES
# ============================================================

print(
    f"\nResults directory: {OUTPUT_DIR.resolve()}"
)

print(
    "\nFloating-point CSV results and displayed tables use "
    f"{SIGNIFICANT_FIGURES} significant figures. "
    "Calculations and ranking use full precision."
)

print(
    "\nRanking penalizes underprediction more heavily, "
    "but model training objectives are unchanged."
)

print(
    "\nGroup-balanced metrics give equal weight to each "
    "Configuration–Model–Year group."
)

print(
    "\nScores are for model development and comparison. "
    "Selecting the best experiment from these scores does "
    "not provide an independent final performance estimate."
)