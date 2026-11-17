from pathlib import Path

import numpy as np
import pandas as pd

from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


# ============================================================
# 0. SETTINGS
# ============================================================

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "5-Regression Data.xlsx"
SHEET_NAME = "Regression Data"

OUTPUT_FILE = DATA_DIR / "resistance_model_data.csv"
REPORT_FILE = DATA_DIR / "data_preparation_report.xlsx"

G = 9.81

# Inclusive lower bound.
MIN_FN = 0.5

GROUP_COLUMNS = [
    "Configuration",
    "Model",
    "Year",
]

GEOMETRY_FEATURES = [
    "Beta",
    "Lp/Bpx",
    "LCG",
    "Ap/Nabla^(2/3)",
    "Ta",
    "Theta_0",
    "CAP",
]

MODEL_FEATURES = GEOMETRY_FEATURES + [
    "Fn",
    "Re",
    "Nabla_Lp3",
    "Trim",
]

TARGET = "C_T"


# ============================================================
# 1. LOAD EXCEL AND RECORD SOURCE UNITS
# ============================================================

units_raw = pd.read_excel(
    INPUT_FILE,
    sheet_name=SHEET_NAME,
    header=None,
    nrows=2,
)

units_rows = []

for name, unit in zip(
    units_raw.iloc[0],
    units_raw.iloc[1],
):
    if pd.notna(name):
        units_rows.append({
            "Column": str(name).strip(),
            "Source unit": (
                str(unit)
                if pd.notna(unit)
                else "—"
            ),
        })

units_df = pd.DataFrame(units_rows)

df = pd.read_excel(
    INPUT_FILE,
    sheet_name=SHEET_NAME,
    header=0,
    skiprows=[1],
)

df.columns = [
    str(column).strip()
    for column in df.columns
]

# Excel layout:
# row 1 = headers
# row 2 = units
# row 3 = first data row
df["source_excel_row"] = np.arange(len(df)) + 3


# ============================================================
# 2. REQUIRED COLUMNS AND NUMERIC CONVERSION
# ============================================================

numeric_columns = list(dict.fromkeys(
    GEOMETRY_FEATURES + [
        "Fn",
        "Lp",
        "Visc",
        "Swet",
        "Nabla",
        "Draft at AP",
        "Draft at FP",
        "Rho",
        "Rtm",
    ]
))

required_columns = numeric_columns + GROUP_COLUMNS

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Preserve original Excel values for the audit report.
source_df = df.copy()

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


# ============================================================
# 3. DERIVED PARAMETERS AND TARGET
# ============================================================

# Fixed dataset definitions:
#
# Fn    : displacement/volume-based Froude number
# Lp    : projected planing-bottom length over chines [m]
# Visc  : kinematic viscosity [m²/s]
# Swet  : running wetted surface area [m²]
# Nabla : displaced volume [m³]
# Draft at AP / FP : running drafts [m]
# Rtm   : measured resistance [N]
# LCG   : longitudinal CG from Ord. 10 [%Lp]
# CAP   : centroid of projected planing area Ap [%Lp]
#
# No automatic unit conversion.
# No ITTC friction calculation.

with np.errstate(
    divide="ignore",
    invalid="ignore",
    over="ignore",
):
    froude_reference_length = np.cbrt(df["Nabla"])

    speed = (
        df["Fn"]
        * np.sqrt(G * froude_reference_length)
    )

    df["Re"] = (
        speed * df["Lp"] / df["Visc"]
    )

    # Original trim convention:
    # positive when aft running draft exceeds forward draft.
    df["Trim"] = np.degrees(
        np.arctan2(
            df["Draft at AP"] - df["Draft at FP"],
            df["Lp"],
        )
    )

    df["Nabla_Lp3"] = (
        df["Nabla"] / df["Lp"]**3
    )

    force_scale = (
        0.5
        * df["Rho"]
        * speed**2
        * df["Swet"]
    )

    df[TARGET] = df["Rtm"] / force_scale


# ============================================================
# 4. VALIDITY CHECKS
# ============================================================

failure_checks = {}


def add_failure(reason, condition):
    """Record True where a row fails a condition."""
    failure_checks[reason] = (
        pd.Series(
            condition,
            index=df.index,
        )
        .fillna(True)
        .astype(bool)
    )


# Source numeric values.
for column in numeric_columns:
    add_failure(
        f"{column}: missing or nonnumeric",
        df[column].isna(),
    )

    add_failure(
        f"{column}: infinite source value",
        np.isinf(
            df[column].to_numpy(dtype=float)
        ),
    )

# Inclusive Froude-number boundary.
add_failure(
    f"Fn: must be at least {MIN_FN}",
    df["Fn"].notna()
    & (df["Fn"] < MIN_FN),
)

for column in [
    "Lp",
    "Visc",
    "Swet",
    "Nabla",
    "Rho",
]:
    add_failure(
        f"{column}: must be positive",
        df[column].notna()
        & (df[column] <= 0),
    )

add_failure(
    "Rtm: must be nonnegative",
    df["Rtm"].notna()
    & (df["Rtm"] < 0),
)

add_failure(
    "Calculated speed: must be finite and positive",
    ~np.isfinite(speed)
    | (speed <= 0),
)

add_failure(
    "Re: must be positive",
    df["Re"].notna()
    & (df["Re"] <= 0),
)

add_failure(
    "Resistance normalization: must be finite and positive",
    ~np.isfinite(force_scale)
    | (force_scale <= 0),
)

for column in MODEL_FEATURES + [TARGET]:
    add_failure(
        f"{column}: non-finite model value",
        ~np.isfinite(
            df[column].to_numpy(dtype=float)
        ),
    )

for column in GROUP_COLUMNS:
    add_failure(
        f"{column}: missing group identifier",
        df[column].isna()
        | df[column].astype(str).str.strip().eq(""),
    )

failure_df = pd.DataFrame(
    failure_checks,
    index=df.index,
)

valid = ~failure_df.any(axis=1)

drop_reasons = failure_df.apply(
    lambda row: "; ".join(
        row.index[
            row.to_numpy(dtype=bool)
        ]
    ),
    axis=1,
)


# ============================================================
# 5. BUILD DROPPED-ROW REPORT TABLES
# ============================================================

dropped_df = source_df.loc[~valid].copy()

dropped_df.insert(
    0,
    "drop_reasons",
    drop_reasons.loc[~valid],
)

# Audit-specific names preserve original source columns.
dropped_df["calculated_speed_m_s"] = speed.loc[~valid]
dropped_df["calculated_Re"] = df.loc[~valid, "Re"]
dropped_df["calculated_Trim_deg"] = df.loc[~valid, "Trim"]
dropped_df["calculated_Nabla_Lp3"] = (
    df.loc[~valid, "Nabla_Lp3"]
)
dropped_df["calculated_C_T"] = df.loc[~valid, TARGET]

leading_columns = [
    "source_excel_row",
    "drop_reasons",
] + GROUP_COLUMNS

remaining_columns = [
    column
    for column in dropped_df.columns
    if column not in leading_columns
]

dropped_df = (
    dropped_df[leading_columns + remaining_columns]
    .sort_values("source_excel_row")
)

reason_summary = (
    failure_df
    .sum()
    .rename_axis("Reason")
    .reset_index(name="Affected rows")
)

reason_summary = (
    reason_summary.loc[
        reason_summary["Affected rows"] > 0
    ]
    .sort_values(
        ["Affected rows", "Reason"],
        ascending=[False, True],
    )
    .reset_index(drop=True)
)


# ============================================================
# 6. BUILD MODELING DATASET
# ============================================================

prepared_df = df.loc[valid].copy()

if not prepared_df.empty:
    group_index = pd.MultiIndex.from_frame(
        prepared_df[GROUP_COLUMNS]
    )

    prepared_df["hull_group"], _ = pd.factorize(
        group_index,
        sort=False,
    )
else:
    prepared_df["hull_group"] = pd.Series(
        index=prepared_df.index,
        dtype="int64",
    )

output_columns = (
    ["source_excel_row"]
    + GROUP_COLUMNS
    + ["hull_group"]
    + MODEL_FEATURES
    + [TARGET]
)

prepared_df = (
    prepared_df[output_columns]
    .reset_index(drop=True)
)

n_groups = prepared_df["hull_group"].nunique()


# ============================================================
# 7. REPORT SUMMARY AND PARAMETER DEFINITIONS
# ============================================================

summary_df = pd.DataFrame(
    [
        ("Input file", str(INPUT_FILE.resolve())),
        ("Source worksheet", SHEET_NAME),
        ("Modeling CSV", str(OUTPUT_FILE.resolve())),
        ("Rows loaded", len(df)),
        ("Rows retained", len(prepared_df)),
        ("Rows removed", int((~valid).sum())),
        ("Groups retained", n_groups),
        ("Fn filter", f"Fn >= {MIN_FN}"),
        ("Froude-number basis", "Displaced volume, Nabla"),
        (
            "Group definition",
            "Configuration–Model–Year",
        ),
        (
            "Modeling CSV precision",
            "Full precision; no explicit rounding",
        ),
        (
            "Exclusion-count interpretation",
            "One row can fail multiple checks; reason counts "
            "may exceed the number of dropped rows.",
        ),
        (
            "Dropped-row contents",
            "Original source values plus calculated audit values",
        ),
        (
            "Source units",
            "Recorded for reference; no automatic conversion",
        ),
        (
            "Friction treatment",
            "No ITTC friction calculation",
        ),
    ],
    columns=["Item", "Value"],
)

definitions_df = pd.DataFrame(
    [
        (
            "Fn",
            "Model input",
            "Displacement-based Froude number",
            "Fn = V / sqrt(g * Nabla^(1/3))",
        ),
        (
            "V",
            "Intermediate only",
            "Reconstructed speed [m/s]",
            "V = Fn * sqrt(g * Nabla^(1/3))",
        ),
        (
            "Lp",
            "Source only",
            "Projected planing-bottom length over chines [m]",
            "Reference length for Re and length ratios",
        ),
        (
            "Visc",
            "Source only",
            "Kinematic viscosity [m²/s]",
            "Used directly in Re",
        ),
        (
            "Re",
            "Model input",
            "Reynolds number based on Lp",
            "Re = V * Lp / Visc",
        ),
        (
            "Nabla",
            "Source only",
            "Displaced volume [m³]",
            "Used for speed and Nabla_Lp3",
        ),
        (
            "Nabla_Lp3",
            "Model input",
            "Volume-to-reference-length-cubed ratio",
            "Nabla / Lp^3",
        ),
        (
            "Trim",
            "Model input",
            "Derived running trim [degrees]; "
            "positive for aft draft greater than forward draft",
            "degrees(arctan2(Draft at AP - Draft at FP, Lp))",
        ),
        (
            "Swet",
            "Source only",
            "Running wetted surface area [m²]",
            "Used to calculate C_T",
        ),
        (
            "Rtm",
            "Source only",
            "Measured resistance [N]",
            "Used to calculate C_T",
        ),
        (
            "Rho",
            "Source only",
            "Fluid density [kg/m³]",
            "Used to calculate C_T",
        ),
        (
            "C_T",
            "Target",
            "Total measured resistance coefficient "
            "normalized using running wetted area",
            "Rtm / (0.5 * Rho * V^2 * Swet)",
        ),
        (
            "LCG",
            "Model input",
            "Longitudinal CG from Ord. 10 [%Lp]",
            "Source percentage representation retained",
        ),
        (
            "CAP",
            "Model input",
            "Centroid of projected planing area Ap [%Lp]",
            "Source percentage representation retained",
        ),
        (
            "Ap/Nabla^(2/3)",
            "Model input",
            "Projected planing area divided by volume^(2/3)",
            "Source value retained",
        ),
        (
            "Beta, Lp/Bpx, Ta, Theta_0",
            "Model inputs",
            "Existing dataset parameters",
            "Source definitions and values retained",
        ),
        (
            "source_excel_row",
            "Metadata",
            "Original worksheet row number",
            "First data row is Excel row 3",
        ),
        (
            "hull_group",
            "Metadata",
            "Integer code for a Configuration–Model–Year tuple",
            "Not a model input",
        ),
    ],
    columns=[
        "Parameter",
        "Role",
        "Definition",
        "Calculation / treatment",
    ],
)

output_schema_df = pd.DataFrame({
    "Column": output_columns,
    "Role": [
        (
            "Target"
            if column == TARGET
            else "Model input"
            if column in MODEL_FEATURES
            else "Metadata"
        )
        for column in output_columns
    ],
})


# ============================================================
# 8. FORMATTED EXCEL REPORT
# ============================================================

report_sheets = {
    "Summary": summary_df,
    "Dropped rows": dropped_df,
    "Exclusion counts": reason_summary,
    "Parameter definitions": definitions_df,
    "Model CSV columns": output_schema_df,
    "Source units": units_df,
}

header_fill = PatternFill(
    fill_type="solid",
    fgColor="17365D",
)
header_font = Font(
    color="FFFFFF",
    bold=True,
)
alternate_fill = PatternFill(
    fill_type="solid",
    fgColor="EEF3F8",
)

with pd.ExcelWriter(
    REPORT_FILE,
    engine="openpyxl",
) as writer:

    for sheet_name, frame in report_sheets.items():
        frame.to_excel(
            writer,
            sheet_name=sheet_name,
            index=False,
            inf_rep="inf",
        )

        worksheet = writer.sheets[sheet_name]

        worksheet.freeze_panes = "A2"
        worksheet.sheet_view.showGridLines = False

        if sheet_name == "Dropped rows":
            # Keep row number and reasons visible while scrolling.
            worksheet.freeze_panes = "C2"

        if not frame.empty:
            worksheet.auto_filter.ref = worksheet.dimensions

        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(
                vertical="center",
                wrap_text=True,
            )

        worksheet.row_dimensions[1].height = 30

        for row in worksheet.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(
                    vertical="top",
                    wrap_text=True,
                )

                if cell.row % 2 == 0:
                    cell.fill = alternate_fill

        # Fit ordinary columns within readable width limits.
        for column_number, column_name in enumerate(
            frame.columns,
            start=1,
        ):
            values = [
                str(column_name),
                *[
                    str(value)
                    for value in frame[column_name]
                    if pd.notna(value)
                ],
            ]

            width = min(
                42,
                max(14, max(map(len, values)) + 2),
            )

            worksheet.column_dimensions[
                get_column_letter(column_number)
            ].width = width

        # Sheet-specific widths for text-heavy content.
        if sheet_name == "Summary":
            worksheet.column_dimensions["A"].width = 30
            worksheet.column_dimensions["B"].width = 95

        elif sheet_name == "Dropped rows":
            worksheet.column_dimensions["A"].width = 20
            worksheet.column_dimensions["B"].width = 85

            for row_number in range(
                2,
                worksheet.max_row + 1,
            ):
                worksheet.row_dimensions[row_number].height = 65

        elif sheet_name == "Exclusion counts":
            worksheet.column_dimensions["A"].width = 85
            worksheet.column_dimensions["B"].width = 18

        elif sheet_name == "Parameter definitions":
            worksheet.column_dimensions["A"].width = 30
            worksheet.column_dimensions["B"].width = 22
            worksheet.column_dimensions["C"].width = 70
            worksheet.column_dimensions["D"].width = 70

            for row_number in range(
                2,
                worksheet.max_row + 1,
            ):
                worksheet.row_dimensions[row_number].height = 45


# ============================================================
# 9. SAVE SEPARATE MODELING CSV
# ============================================================

# Write the report even if no rows remain, but do not replace
# the modeling CSV with an empty dataset.
if prepared_df.empty:
    raise ValueError(
        "No usable rows remain. "
        f"Review {REPORT_FILE.resolve()}. "
        "No new modeling CSV was written."
    )

# Full precision: no float_format or explicit rounding.
prepared_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# 10. DISPLAY SUMMARY
# ============================================================

print(f"Rows loaded:   {len(df)}")
print(f"Rows retained: {len(prepared_df)}")
print(f"Rows removed:  {int((~valid).sum())}")
print(f"Groups retained: {n_groups}")

print(f"\nFn filter: Fn >= {MIN_FN}")
print(
    "Speed calculation: "
    "V = Fn * sqrt(g * Nabla^(1/3))"
)

print(
    "\nDropped original Excel rows:",
    dropped_df["source_excel_row"].tolist(),
)

print(
    f"\nModeling data:\n{OUTPUT_FILE.resolve()}"
)
print(
    f"\nPreparation report:\n{REPORT_FILE.resolve()}"
)

print(
    "\nNext, run run_resistance_models.py."
)