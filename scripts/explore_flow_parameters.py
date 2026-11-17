from pathlib import Path
from datetime import datetime
import json
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    PolynomialFeatures,
)
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


# ============================================================
# 0. SETTINGS
# ============================================================

# Paths are relative to the current working directory.
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs") / "flow_exploration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "resistance_model_data.csv"

TARGET = "C_T"

N_SPLITS = 5
RANDOM_STATE = 42

UNDERPRED_WEIGHT = 2.0
RANKING_METRIC = "GroupBalanced_AsymMAE"

SIGNIFICANT_FIGURES = 4
CSV_FLOAT_FORMAT = f"%.{SIGNIFICANT_FIGURES}g"

# Fixed Ridge screening-model configuration.
#
# Only the hull/static-condition block receives automatic
# polynomial expansion.
#
# Flow terms are explicitly constructed and are NOT expanded
# again or automatically crossed with the hull block.
HULL_POLYNOMIAL_DEGREE = 2
RIDGE_ALPHA = 10.0

# Fixed histogram gradient boosting configuration.
HISTGB_PARAMETERS = {
    "loss": "squared_error",
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
    "early_stopping": False,
    "random_state": RANDOM_STATE,
}

# Reference representation for within-model comparisons.
BASELINE_FLOW = "Fn_Linear__Re_Linear"

TOP_N_DISPLAY = 15
TOP_N_PLOT = 15

if UNDERPRED_WEIGHT < 1.0:
    raise ValueError(
        "UNDERPRED_WEIGHT must be at least 1."
    )

if N_SPLITS < 2:
    raise ValueError(
        "N_SPLITS must be at least 2."
    )


# ============================================================
# 1. OUTPUT HELPERS
# ============================================================

def format_float(value):
    """Format floating-point output to four significant figures."""
    return f"{value:.{SIGNIFICANT_FIGURES}g}"


def save_csv(frame, filename):
    """
    Save results with formatted floating-point values.

    All calculations and ranking take place before formatting.
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
    return format_float(value)


# ============================================================
# 2. READ AND VALIDATE PREPARED DATA
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Prepared dataset not found:\n{INPUT_FILE.resolve()}\n"
        "Run the preparation script first."
    )

df = pd.read_csv(INPUT_FILE)

# These inputs remain unchanged in every flow experiment.
#
# Ta:
#   Deadrise at Ord. 10 minus deadrise at Ord. 0.
#
# Theta_0:
#   Zero-speed/static trim, positive bow-down.
#
# LCG:
#   Longitudinal CG from Ord. 10 [%Lp].
#
# CAP:
#   Centroid of projected planing area Ap [%Lp].
#
# LCG and CAP retain their source percentage representation.
#
# Measured running Trim is deliberately excluded.
#
# Static wetted area is not included because it is not yet
# present in the prepared-data schema.

FIXED_FEATURES = [
    "Beta",
    "Lp/Bpx",
    "LCG",
    "Ap/Nabla^(2/3)",
    "Ta",
    "Theta_0",
    "CAP",
    "Nabla_Lp3",
]

METADATA_COLUMNS = [
    "source_excel_row",
    "Configuration",
    "Model",
    "Year",
    "hull_group",
]

required_columns = list(dict.fromkeys(
    METADATA_COLUMNS
    + FIXED_FEATURES
    + ["Fn", "Re", TARGET]
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

if df[METADATA_COLUMNS].isna().any().any():
    raise ValueError(
        "Prepared CSV contains missing identifiers."
    )

numeric_columns = list(dict.fromkeys(
    FIXED_FEATURES + ["Fn", "Re", TARGET]
))

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="raise",
    )

if not np.isfinite(
    df[numeric_columns].to_numpy(dtype=float)
).all():
    raise ValueError(
        "Prepared CSV contains non-finite modeling values."
    )

# All variants use the same rows.
# Logs and reciprocals require strictly positive flow inputs.
for column in ["Fn", "Re"]:
    if (df[column] <= 0).any():
        raise ValueError(
            f"All {column} values must be positive."
        )

if (df[TARGET] < 0).any():
    raise ValueError(
        "The prepared C_T target contains negative values."
    )

if df["source_excel_row"].duplicated().any():
    raise ValueError(
        "Duplicate source_excel_row identifiers found."
    )

# Confirm that hull_group codes agree with the intended
# Configuration–Model–Year grouping.
group_check = df[
    ["hull_group", "Configuration", "Model", "Year"]
].drop_duplicates()

if group_check["hull_group"].duplicated().any():
    raise ValueError(
        "A hull_group code maps to multiple identifier tuples."
    )

if group_check.duplicated(
    ["Configuration", "Model", "Year"]
).any():
    raise ValueError(
        "An identifier tuple maps to multiple hull_group codes."
    )

n_groups = df["hull_group"].nunique()

if n_groups < N_SPLITS:
    raise ValueError(
        f"{N_SPLITS} folds requested, but only "
        f"{n_groups} groups are available."
    )

df = df.reset_index(drop=True)

y = df[TARGET].to_numpy(dtype=float)
groups = df["hull_group"].to_numpy()


# ============================================================
# 3. BUILD THE TWELVE REPRESENTATIONS FOR EACH FLOW INPUT
# ============================================================

VARIANT_ORDER = [
    "Zero",
    "Linear",
    "Quadratic",
    "Cubic",
    "Log",
    "LogQuadratic",
    "LogCubic",
    "Root",
    "Reciprocal",
    "RawPlusLog",
    "RawPlusRoot",
    "RawPlusReciprocal",
]


def add_flow_representations(frame, source_column):
    """
    Add explicit row-wise flow transformations.

    Zero means omit the source parameter from the predictors.
    It does not set its physical value to zero or recalculate C_T.

    Quadratic/cubic representations use cumulative powers.
    Natural logarithms are used throughout.
    """
    x = frame[source_column].to_numpy(dtype=float)

    prefix = f"flow_{source_column}"

    names = {
        "raw": f"{prefix}_raw",
        "square": f"{prefix}_square",
        "cube": f"{prefix}_cube",
        "log": f"{prefix}_ln",
        "log_square": f"{prefix}_ln_square",
        "log_cube": f"{prefix}_ln_cube",
        "root": f"{prefix}_root",
        "reciprocal": f"{prefix}_reciprocal",
    }

    with np.errstate(
        divide="ignore",
        invalid="ignore",
        over="ignore",
    ):
        log_x = np.log(x)

        frame[names["raw"]] = x
        frame[names["square"]] = x**2
        frame[names["cube"]] = x**3

        frame[names["log"]] = log_x
        frame[names["log_square"]] = log_x**2
        frame[names["log_cube"]] = log_x**3

        frame[names["root"]] = np.sqrt(x)
        frame[names["reciprocal"]] = 1.0 / x

    generated_columns = list(names.values())

    if not np.isfinite(
        frame[generated_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            f"Non-finite transformed values generated for "
            f"{source_column}."
        )

    variants = {
        "Zero": [],

        "Linear": [
            names["raw"],
        ],

        "Quadratic": [
            names["raw"],
            names["square"],
        ],

        "Cubic": [
            names["raw"],
            names["square"],
            names["cube"],
        ],

        "Log": [
            names["log"],
        ],

        "LogQuadratic": [
            names["log"],
            names["log_square"],
        ],

        "LogCubic": [
            names["log"],
            names["log_square"],
            names["log_cube"],
        ],

        "Root": [
            names["root"],
        ],

        "Reciprocal": [
            names["reciprocal"],
        ],

        "RawPlusLog": [
            names["raw"],
            names["log"],
        ],

        "RawPlusRoot": [
            names["raw"],
            names["root"],
        ],

        "RawPlusReciprocal": [
            names["raw"],
            names["reciprocal"],
        ],
    }

    return variants


fn_variants = add_flow_representations(df, "Fn")
re_variants = add_flow_representations(df, "Re")

flow_configurations = []

for fn_form in VARIANT_ORDER:
    for re_form in VARIANT_ORDER:
        flow_features = (
            fn_variants[fn_form]
            + re_variants[re_form]
        )

        flow_configurations.append({
            "name": f"Fn_{fn_form}__Re_{re_form}",
            "Fn_form": fn_form,
            "Re_form": re_form,
            "flow_features": flow_features,
        })


# ============================================================
# 4. FIXED SCREENING MODEL BUILDERS
# ============================================================

RIDGE_MODEL_NAME = "HullPoly2_Ridge_a10"
HISTGB_MODEL_NAME = "HistGB_leaves15_l2_0"

MODEL_NAMES = [
    RIDGE_MODEL_NAME,
    HISTGB_MODEL_NAME,
]


def build_ridge_model(flow_features):
    """
    Fixed quadratic hull/static block plus explicit flow basis.

    No automatic flow powers, flow-flow interactions, or
    flow-hull interactions are added.

    All learned preprocessing is fitted within each training fold.
    """
    hull_pipeline = Pipeline([
        ("input_scale", StandardScaler()),
        (
            "polynomial",
            PolynomialFeatures(
                degree=HULL_POLYNOMIAL_DEGREE,
                include_bias=False,
            ),
        ),
        ("term_scale", StandardScaler()),
    ])

    transformers = [
        (
            "hull",
            hull_pipeline,
            FIXED_FEATURES,
        ),
    ]

    if flow_features:
        transformers.append(
            (
                "flow",
                StandardScaler(),
                flow_features,
            )
        )

    preprocessing = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0,
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        (
            "model",
            Ridge(alpha=RIDGE_ALPHA),
        ),
    ])


def build_screening_model(model_name, flow_features):
    if model_name == RIDGE_MODEL_NAME:
        return build_ridge_model(flow_features)

    if model_name == HISTGB_MODEL_NAME:
        return HistGradientBoostingRegressor(
            **HISTGB_PARAMETERS
        )

    raise ValueError(
        f"Unknown model name: {model_name}"
    )


# ============================================================
# 5. FIXED GROUPED CROSS-VALIDATION SPLITS
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

fold_design_rows = []

for fold, (train_idx, test_idx) in enumerate(
    splits,
    start=1,
):
    fold_assignment[test_idx] = fold

    # Defensive check: no group may cross the split.
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])

    if train_groups.intersection(test_groups):
        raise RuntimeError(
            f"Group overlap detected in fold {fold}."
        )

    fold_design_rows.append({
        "Fold": fold,
        "N_train": len(train_idx),
        "N_test": len(test_idx),
        "N_train_groups": len(train_groups),
        "N_test_groups": len(test_groups),
    })

assignments = df[METADATA_COLUMNS].copy()
assignments["fold"] = fold_assignment

save_csv(
    assignments,
    "flow_cv_assignments.csv",
)

save_csv(
    pd.DataFrame(fold_design_rows),
    "flow_fold_design.csv",
)


# ============================================================
# 6. METRICS
# ============================================================

def calculate_metrics(y_true, y_pred, group_ids):
    """
    Bias convention:
        predicted - measured

    Asymmetric absolute error:
        UNDERPRED_WEIGHT * shortfall + overprediction excess

    Group-balanced metrics weight each group equally.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    group_ids = np.asarray(group_ids)

    error = y_pred - y_true
    absolute_error = np.abs(error)

    shortfall = np.maximum(-error, 0.0)
    excess = np.maximum(error, 0.0)
    under = error < 0

    asymmetric_error = (
        UNDERPRED_WEIGHT * shortfall
        + excess
    )

    per_group = (
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
        "GroupBalanced_AsymMAE": (
            per_group["asymmetric_error"].mean()
        ),

        "GroupBalanced_MAE": (
            per_group["absolute_error"].mean()
        ),

        "AsymMAE": asymmetric_error.mean(),

        "MAE": mean_absolute_error(
            y_true,
            y_pred,
        ),

        "RMSE": np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        ),

        "Bias": error.mean(),

        "Underpred_Freq_pct": (
            100.0 * under.mean()
        ),

        "Underpred_Mag": (
            shortfall[under].mean()
            if under.any()
            else 0.0
        ),

        "MeanShortfall": shortfall.mean(),

        "NegativePrediction_pct": (
            100.0 * np.mean(y_pred < 0)
        ),
    }


# ============================================================
# 7. SAVE EXPERIMENT CONFIGURATION
# ============================================================

n_candidates = (
    len(flow_configurations)
    * len(MODEL_NAMES)
)

n_fits = n_candidates * N_SPLITS

configuration = {
    "input_file": str(INPUT_FILE.resolve()),
    "output_directory": str(OUTPUT_DIR.resolve()),

    "target": TARGET,
    "n_rows": len(df),
    "n_groups": int(n_groups),
    "n_splits": N_SPLITS,

    "n_flow_configurations": len(flow_configurations),
    "n_candidates": n_candidates,
    "n_model_fits": n_fits,

    "random_state": RANDOM_STATE,
    "underprediction_weight": UNDERPRED_WEIGHT,
    "ranking_metric": RANKING_METRIC,
    "output_significant_figures": SIGNIFICANT_FIGURES,

    "baseline_flow": BASELINE_FLOW,
    "fixed_features": FIXED_FEATURES,
    "variant_order": VARIANT_ORDER,

    "Fn_variants": fn_variants,
    "Re_variants": re_variants,

    "models": {
        RIDGE_MODEL_NAME: {
            "hull_polynomial_degree": HULL_POLYNOMIAL_DEGREE,
            "ridge_alpha": RIDGE_ALPHA,
            "automatic_flow_expansion": False,
            "explicit_flow_hull_interactions": False,
        },
        HISTGB_MODEL_NAME: HISTGB_PARAMETERS,
    },

    "notes": [
        "Zero means omitted predictor, not zero physical flow.",
        "Fn is displacement/volume-based.",
        "Prepared targets are not recalculated.",
        "Measured running Trim is not an input.",
        "Theta_0 remains a static-condition input.",
        "Static wetted area is not yet present in the input schema.",
        "All candidates use the same rows and grouped folds.",
        "Ridge uses a fixed quadratic hull/static block.",
        "Flow basis terms are explicitly supplied without "
        "automatic polynomial re-expansion.",
        "Histogram boosting can learn interactions through its trees.",
        "Training objectives are symmetric; ranking is asymmetric.",
        "Results are development comparisons, not an independent "
        "final performance estimate.",
    ],
}

with open(
    OUTPUT_DIR / "flow_experiment_config.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        configuration,
        file,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# 8. RUN ALL CANDIDATES
# ============================================================

print(f"Input: {INPUT_FILE.resolve()}")
print(f"Outputs: {OUTPUT_DIR.resolve()}")
print(f"Target: {TARGET}")
print(f"Rows: {len(df)}")
print(f"Groups: {n_groups}")

print(
    f"\nFlow configurations: {len(flow_configurations)}"
)
print(f"Screening models: {len(MODEL_NAMES)}")
print(f"Candidates: {n_candidates}")
print(f"Individual cross-validation fits: {n_fits}")

print(
    "\nMeasured running Trim is excluded from predictors."
)
print(
    "Zero means omission of that flow parameter."
)

summary_rows = []
fold_rows = []

candidate_number = 0

# Stream predictions to disk instead of retaining every
# candidate's predictions in memory.
oof_path = OUTPUT_DIR / "flow_oof_predictions.csv"
first_prediction_write = True

run_start = time.perf_counter()

for model_name in MODEL_NAMES:
    for flow in flow_configurations:
        candidate_number += 1

        experiment_name = (
            f"{model_name}__{flow['name']}"
        )

        print(
            f"[{candidate_number}/{n_candidates}] "
            f"{experiment_name}"
        )

        candidate_start = time.perf_counter()

        input_features = (
            FIXED_FEATURES
            + flow["flow_features"]
        )

        X = df[input_features]

        base_estimator = build_screening_model(
            model_name,
            flow["flow_features"],
        )

        oof_prediction = np.full(
            len(df),
            np.nan,
        )

        fold_scores = []

        for fold, (train_idx, test_idx) in enumerate(
            splits,
            start=1,
        ):
            estimator = clone(base_estimator)

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

            # Keep predictions unmodified: no clipping.
            oof_prediction[test_idx] = prediction

            fold_metrics = calculate_metrics(
                y[test_idx],
                prediction,
                groups[test_idx],
            )

            fold_scores.append(
                fold_metrics[RANKING_METRIC]
            )

            fold_rows.append({
                "Experiment": experiment_name,
                "Model": model_name,
                "Flow": flow["name"],
                "Fn_form": flow["Fn_form"],
                "Re_form": flow["Re_form"],
                "Fold": fold,
                "N_train": len(train_idx),
                "N_test": len(test_idx),
                **fold_metrics,
            })

        if not np.isfinite(oof_prediction).all():
            raise RuntimeError(
                f"Incomplete predictions: {experiment_name}"
            )

        # Main scores are calculated from pooled out-of-fold
        # predictions, with group balancing applied explicitly.
        metrics = calculate_metrics(
            y,
            oof_prediction,
            groups,
        )

        elapsed = (
            time.perf_counter() - candidate_start
        )

        summary_rows.append({
            "Experiment": experiment_name,
            "Model": model_name,
            "Flow": flow["name"],
            "Fn_form": flow["Fn_form"],
            "Re_form": flow["Re_form"],
            "Target": TARGET,

            # Count before Ridge's fixed hull expansion.
            "N_input_features": len(input_features),
            "N_flow_features": len(flow["flow_features"]),

            "N_rows": len(df),
            "N_groups": n_groups,

            **metrics,

            "Fold_Score_mean": np.mean(fold_scores),
            "Fold_Score_std": np.std(
                fold_scores,
                ddof=1,
            ),

            "Elapsed_seconds": elapsed,
        })

        prediction_frame = assignments.copy()

        prediction_frame["Experiment"] = experiment_name
        prediction_frame["Model_name"] = model_name
        prediction_frame["Flow"] = flow["name"]

        prediction_frame["Fn"] = df["Fn"].to_numpy()
        prediction_frame["Re"] = df["Re"].to_numpy()

        prediction_frame["C_T_true"] = y
        prediction_frame["C_T_pred"] = oof_prediction

        prediction_frame.to_csv(
            oof_path,
            mode=(
                "w"
                if first_prediction_write
                else "a"
            ),
            header=first_prediction_write,
            index=False,
            float_format=CSV_FLOAT_FORMAT,
        )

        first_prediction_write = False

        # Save progress after each completed candidate.
        # These are replaced by finalized tables below.
        save_csv(
            pd.DataFrame(summary_rows),
            "flow_summary.csv",
        )

        save_csv(
            pd.DataFrame(fold_rows),
            "flow_fold_results.csv",
        )


# ============================================================
# 9. FINALIZE WITHIN-MODEL RANKINGS
# ============================================================

summary_df = pd.DataFrame(summary_rows)

baseline_scores = (
    summary_df.loc[
        summary_df["Flow"] == BASELINE_FLOW
    ]
    .set_index("Model")[RANKING_METRIC]
)

summary_df["Baseline_score"] = (
    summary_df["Model"].map(baseline_scores)
)

summary_df["Improvement_vs_raw_pct"] = np.where(
    summary_df["Baseline_score"] > 0,
    (
        100.0
        * (
            summary_df["Baseline_score"]
            - summary_df[RANKING_METRIC]
        )
        / summary_df["Baseline_score"]
    ),
    np.nan,
)

# Rankings are calculated before output rounding.
summary_df = (
    summary_df
    .sort_values(
        ["Model", RANKING_METRIC, "RMSE"],
        ascending=True,
    )
    .reset_index(drop=True)
)

summary_df["Rank_within_model"] = (
    summary_df.groupby("Model").cumcount() + 1
)

leading_columns = [
    "Model",
    "Rank_within_model",
    "Flow",
    "Fn_form",
    "Re_form",
    "Improvement_vs_raw_pct",
]

summary_df = summary_df[
    leading_columns
    + [
        column
        for column in summary_df.columns
        if column not in leading_columns
    ]
]

fold_df = pd.DataFrame(fold_rows)

score_comparison = summary_df.pivot(
    index="Flow",
    columns="Model",
    values=RANKING_METRIC,
)

improvement_comparison = summary_df.pivot(
    index="Flow",
    columns="Model",
    values="Improvement_vs_raw_pct",
)

save_csv(
    summary_df,
    "flow_summary.csv",
)

save_csv(
    fold_df,
    "flow_fold_results.csv",
)

save_csv(
    score_comparison.reset_index(),
    "flow_score_comparison.csv",
)

save_csv(
    improvement_comparison.reset_index(),
    "flow_improvement_comparison.csv",
)


# ============================================================
# 10. DISPLAY LEADING CANDIDATES
# ============================================================

display_columns = [
    "Rank_within_model",
    "Flow",
    RANKING_METRIC,
    "Improvement_vs_raw_pct",
    "GroupBalanced_MAE",
    "RMSE",
    "Bias",
    "Underpred_Freq_pct",
    "NegativePrediction_pct",
]

for model_name in MODEL_NAMES:
    model_results = summary_df.loc[
        summary_df["Model"] == model_name
    ]

    print(
        f"\nLeading candidates: {model_name}"
    )

    print(
        model_results[display_columns]
        .head(TOP_N_DISPLAY)
        .to_string(
            index=False,
            float_format=format_float,
        )
    )


# ============================================================
# 11. TOP-CANDIDATE FIGURE
# ============================================================

timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S_%f"
)

fig, axes = plt.subplots(
    1,
    len(MODEL_NAMES),
    figsize=(20, 9),
    squeeze=False,
)

for ax, model_name in zip(
    axes.ravel(),
    MODEL_NAMES,
):
    plot_df = (
        summary_df.loc[
            summary_df["Model"] == model_name
        ]
        .head(TOP_N_PLOT)
        .iloc[::-1]
    )

    colors = [
        "darkorange"
        if flow_name == BASELINE_FLOW
        else "steelblue"
        for flow_name in plot_df["Flow"]
    ]

    ax.barh(
        plot_df["Flow"],
        plot_df[RANKING_METRIC],
        color=colors,
    )

    ax.set_title(model_name)

    ax.set_xlabel(
        "Group-balanced asymmetric MAE\n"
        f"(underprediction weight = "
        f"{format_float(UNDERPRED_WEIGHT)})"
    )

    ax.tick_params(
        axis="y",
        labelsize=8,
    )

    ax.xaxis.set_major_formatter(
        FuncFormatter(format_plot_axis)
    )

fig.suptitle(
    "Leading flow representations by screening model\n"
    "Orange marks the raw Fn + raw Re reference if shown",
    fontsize=13,
)

fig.tight_layout()

top_figure_path = (
    OUTPUT_DIR.resolve()
    / f"flow_top_candidates_{timestamp}.png"
)

try:
    fig.savefig(
        str(top_figure_path),
        format="png",
        dpi=150,
        bbox_inches="tight",
    )

    print(
        f"\nTop-candidate figure saved:\n"
        f"{top_figure_path}"
    )

except OSError as exc:
    print(
        f"\nTop-candidate figure could not be saved: {exc}"
    )

finally:
    plt.close(fig)


# ============================================================
# 12. FULL FLOW-GRID HEATMAPS
# ============================================================

fig, axes = plt.subplots(
    1,
    len(MODEL_NAMES),
    figsize=(21, 10),
    squeeze=False,
)

for ax, model_name in zip(
    axes.ravel(),
    MODEL_NAMES,
):
    model_results = summary_df.loc[
        summary_df["Model"] == model_name
    ]

    grid = (
        model_results.pivot(
            index="Fn_form",
            columns="Re_form",
            values=RANKING_METRIC,
        )
        .reindex(
            index=VARIANT_ORDER,
            columns=VARIANT_ORDER,
        )
    )

    image = ax.imshow(
        grid.to_numpy(dtype=float),
        cmap="viridis_r",
        aspect="auto",
    )

    ax.set_xticks(
        np.arange(len(VARIANT_ORDER))
    )
    ax.set_yticks(
        np.arange(len(VARIANT_ORDER))
    )

    ax.set_xticklabels(
        VARIANT_ORDER,
        rotation=60,
        ha="right",
    )
    ax.set_yticklabels(
        VARIANT_ORDER
    )

    ax.set_xlabel("Re representation")
    ax.set_ylabel("Fn representation")
    ax.set_title(model_name)

    # Mark the best configuration for this screening model.
    best_row = model_results.iloc[0]

    best_x = VARIANT_ORDER.index(
        best_row["Re_form"]
    )
    best_y = VARIANT_ORDER.index(
        best_row["Fn_form"]
    )

    ax.scatter(
        best_x,
        best_y,
        marker="x",
        s=110,
        color="red",
        linewidths=2,
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    colorbar.set_label(
        "Group-balanced asymmetric MAE"
    )
    colorbar.formatter = FuncFormatter(
        format_plot_axis
    )
    colorbar.update_ticks()

fig.suptitle(
    "All 144 flow representations per model\n"
    "Lower scores are better; red cross marks the minimum. "
    "Each panel has its own color scale.",
    fontsize=13,
)

fig.tight_layout()

heatmap_path = (
    OUTPUT_DIR.resolve()
    / f"flow_grid_{timestamp}.png"
)

try:
    fig.savefig(
        str(heatmap_path),
        format="png",
        dpi=150,
        bbox_inches="tight",
    )

    print(
        f"\nFlow-grid figure saved:\n"
        f"{heatmap_path}"
    )

except OSError as exc:
    print(
        f"\nFlow-grid figure could not be saved: {exc}"
    )

finally:
    plt.close(fig)


# ============================================================
# 13. COMPLETION
# ============================================================

total_elapsed = (
    time.perf_counter() - run_start
)

print(
    "\nTotal elapsed seconds:",
    format_float(total_elapsed),
)

print(
    f"\nResults directory:\n"
    f"{OUTPUT_DIR.resolve()}"
)

print(
    "\nPositive Improvement_vs_raw_pct means lower error "
    "than raw Fn + raw Re for the same screening model."
)

print(
    "\nCandidates are ranked separately for each model. "
    "Scores are not averaged across model families."
)

print(
    "\nFloating-point outputs use four significant figures. "
    "Model fitting, metric calculation, and ranking use "
    "full precision."
)

print(
    "\nThe models use symmetric training objectives. "
    "Underprediction is penalized more heavily during ranking."
)

print(
    "\nThese are development results. Selecting a feature "
    "representation from these scores does not provide an "
    "independent final estimate of predictive performance."
)