import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from load_data import load_data

# --------------------------------------------------------
# Chart Folder
# --------------------------------------------------------

CHARTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "static",
    "charts"
)

os.makedirs(CHARTS_DIR, exist_ok=True)

sns.set_style("whitegrid")
sns.set_context("paper")


def _chart_path(filename):
    return os.path.join(CHARTS_DIR, filename)


def _save(filename):
    plt.tight_layout()
    plt.savefig(
        _chart_path(filename),
        dpi=120,
        bbox_inches="tight"
    )
    plt.close()

# MAIN EDA FUNCTION
def run_eda():
    data = load_data()
    charts = []
    # 1. DATASET SUMMARY
    rows = len(data)
    columns = len(data.columns)
    basic_info = pd.DataFrame({
        "Data Type": data.dtypes.astype(str),
        "Non Null Count": data.count()
    })

    numeric_summary = (
        data.describe()
        .round(2)
        .to_dict()
    )

    categorical_summary = (
        data.describe(
            include=["object", "string"]
        )
        .fillna("")
        .to_dict()
    )

    # 2. BASIC INFORMATION
    dtypes = data.dtypes.astype(str).to_dict()

    # 3. MISSING VALUES
    missing = data.isnull().sum()
    missing_pct = (
        missing / len(data)
    ) * 100

    missing_df = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": missing_pct
    })

    missing_df = missing_df[
        missing_df["missing_count"] > 0
    ].sort_values(
        "missing_count",
        ascending=False
    )

    if not missing_df.empty:

        plt.figure(figsize=(8,4))

        sns.barplot(
            x=missing_df.index,
            y=missing_df["missing_pct"]
        )

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.ylabel("Missing %")
        plt.title("Missing Values by Column")

        _save("missing_values.png")

        charts.append(
            "charts/missing_values.png"
        )

        plt.figure(figsize=(8,4))

        sns.heatmap(
            data.isnull(),
            cmap="viridis",
            cbar=False
        )

        plt.title("Missing Values Heatmap")

        _save("missing_heatmap.png")

        charts.append(
            "charts/missing_heatmap.png"
        )

    # ====================================================
    # 4. DUPLICATE ROWS


    duplicates = int(
        data.duplicated().sum()
    )
    # 5. TARGET VARIABLE
    target_counts = (
        data["PlacementStatus"]
        .value_counts()
        .to_dict()
    )

    plt.figure(figsize=(5,4))

    sns.countplot(
        x="PlacementStatus",
        data=data
    )

    plt.title(
        "Placement Status Distribution"
    )

    plt.xlabel(
        "Placement Status"
    )

    plt.ylabel("Count")

    _save(
        "placement_status.png"
    )

    charts.append(
        "charts/placement_status.png"
    )

    # ====================================================
    # 6. NUMERIC FEATURE DISTRIBUTION
    # ====================================================

    hist_cols = [
        "CGPA",
        "AttendancePercentage",
        "AptitudeTestScore",
        "SoftSkillsRating",
        "CodingTestScore",
        "MockInterviewScore",
        "Projects",
        "Workshops",
        "PlacementStatus",
        "Salary Package",
        "SGPA_Sem1",
        "SGPA_Sem2",
        "SGPA_Sem3",
        "SGPA_Sem4",
        "SGPA_Sem5"
    ]

    hist_cols = [c for c in hist_cols if c in data.columns]

    if hist_cols:

        data[hist_cols].hist(
            figsize=(14,10),
            bins=20
        )

        _save("numeric_distribution.png")

        charts.append(
            "charts/numeric_distribution.png"
        )

    if "CGPA" in data.columns:

        plt.figure(figsize=(6,4))

        sns.histplot(
            data["CGPA"],
            kde=True
        )

        plt.axvline(
            data["CGPA"].mean(),
            color="green",
            linestyle="--",
            label="Mean"
        )

        plt.legend()

        plt.title(
            "CGPA Distribution"
        )

        _save("cgpa_distribution.png")

        charts.append(
            "charts/cgpa_distribution.png"
        )

    # ====================================================
    # 7. OUTLIER DETECTION
    # ====================================================

    box_cols = [
        "CGPA",
        "AttendancePercentage",
        "AptitudeTestScore",
        "SoftSkillsRating",
        "CodingTestScore",
        "MockInterviewScore",
        "Projects"
    ]

    box_cols = [
        c for c in box_cols
        if c in data.columns
    ]

    for col in box_cols:

        plt.figure(figsize=(7,4))

        sns.boxplot(
            x=data[col],
            color="gold"
        )

        plt.title(
            f"Boxplot - {col}"
        )

        filename = f"boxplot_{col}.png"

        _save(filename)

        charts.append(
            f"charts/{filename}"
        )

    # ====================================================
    # 8. CORRELATION HEATMAP
    # ====================================================

    corr = (
        data
        .select_dtypes(include=[np.number])
        .corr()
        .round(2)
    )

    correlation_table = corr.to_dict()

    plt.figure(figsize=(12,8))

    sns.heatmap(
        corr,
        annot=True,
        cmap="Pastel2",
        fmt=".2f"
    )

    plt.title(
        "Correlation Heatmap"
    )

    _save(
        "correlation_heatmap.png"
    )

    charts.append(
        "charts/correlation_heatmap.png"
    )

    # ====================================================
    # 9. REGRESSION SCATTER PLOT
    # ====================================================

    if (
        "CGPA" in data.columns and
        "Salary Package" in data.columns
    ):

        plt.figure(figsize=(7,4))

        sns.regplot(
            x="CGPA",
            y="Salary Package",
            data=data,
            color="skyblue"
        )

        plt.title(
            "CGPA vs Salary Package"
        )

        _save(
            "cgpa_salary_regression.png"
        )

        charts.append(
            "charts/cgpa_salary_regression.png"
        )

    # ====================================================
    # 10. CATEGORICAL FEATURE COUNTS
    # ====================================================

    cat_cols = [

        "Gender",

        "City",

        "CollegeTier",

        "Stream",

        "Specialisation",

        "Hostel",

        "HistoryOfBacklogs",

        "CGPA_Tier"

    ]

    cat_cols = [
        c for c in cat_cols
        if c in data.columns
    ]

    categorical_counts = {}

    for col in cat_cols:

        categorical_counts[col] = (
            data[col]
            .value_counts()
            .to_dict()
        )

        plt.figure(figsize=(8,4))

        sns.countplot(
            x=col,
            data=data
        )

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.title(
            f"{col} Distribution"
        )

        filename = f"{col.lower()}_distribution.png"

        _save(filename)

        charts.append(
            f"charts/{filename}"
        )

        # ====================================================
        # 11. GENDER vs PLACEMENT STATUS
        # ====================================================

        if (
                "Gender" in data.columns and
                "PlacementStatus" in data.columns
        ):

            gender_placement = pd.crosstab(
                data["Gender"],
                data["PlacementStatus"]
            )

            plt.figure(figsize=(7, 4))

            sns.countplot(
                x="Gender",
                hue="PlacementStatus",
                data=data
            )

            plt.title("Gender vs Placement Status")

            _save("gender_vs_placement.png")

            charts.append(
                "charts/gender_vs_placement.png"
            )

        else:
            gender_placement = {}

        # ====================================================
        # 12. COLLEGE TIER / STREAM vs PLACEMENT STATUS
        # ====================================================

        if (
                "CollegeTier" in data.columns and
                "PlacementStatus" in data.columns
        ):

            tier_placement = pd.crosstab(
                data["CollegeTier"],
                data["PlacementStatus"]
            )

            plt.figure(figsize=(7, 4))

            sns.countplot(
                x="CollegeTier",
                hue="PlacementStatus",
                data=data
            )

            plt.title("College Tier vs Placement")

            _save("tier_vs_placement.png")

            charts.append(
                "charts/tier_vs_placement.png"
            )

        else:
            tier_placement = {}

        if (
                "Stream" in data.columns and
                "PlacementStatus" in data.columns
        ):

            stream_placement = pd.crosstab(
                data["Stream"],
                data["PlacementStatus"]
            )

            plt.figure(figsize=(10, 5))

            sns.countplot(
                x="Stream",
                hue="PlacementStatus",
                data=data
            )

            plt.xticks(
                rotation=45,
                ha="right"
            )

            plt.title("Stream vs Placement")

            _save("stream_vs_placement.png")

            charts.append(
                "charts/stream_vs_placement.png"
            )

        else:
            stream_placement = {}


        # 13. SGPA TREND

        sgpa_cols = [
            f"SGPA_Sem{i}"
            for i in range(1, 9)
            if f"SGPA_Sem{i}" in data.columns
        ]

        avg_sgpa = {}

        if sgpa_cols:
            avg_sgpa = (
                data[sgpa_cols]
                .mean()
                .round(2)
                .to_dict()
            )

            plt.figure(figsize=(8, 4))

            plt.plot(
                list(avg_sgpa.keys()),
                list(avg_sgpa.values()),
                marker="o"
            )

            plt.grid(True)

            plt.title(
                "Average SGPA Trend"
            )

            plt.xlabel("Semester")

            plt.ylabel("Average SGPA")

            _save("sgpa_trend.png")

            charts.append(
                "charts/sgpa_trend.png"
            )

        # ====================================================
        # 14. SALARY PACKAGE ANALYSIS
        # ====================================================

        salary_stats = {}

        if (
                "PlacementStatus" in data.columns and
                "Salary Package" in data.columns
        ):

            placed_data = data[
                data["PlacementStatus"] == "Placed"
                ]

            if not placed_data.empty:

                salary_stats = (
                    placed_data["Salary Package"]
                    .describe()
                    .round(2)
                    .to_dict()
                )

                plt.figure(figsize=(7, 4))

                sns.histplot(
                    placed_data["Salary Package"],
                    bins=20,
                    kde=True
                )

                plt.title(
                    "Salary Distribution"
                )

                _save(
                    "salary_distribution.png"
                )

                charts.append(
                    "charts/salary_distribution.png"
                )

                if "CollegeTier" in placed_data.columns:

                    salary_by_tier = (
                        placed_data
                        .groupby("CollegeTier")["Salary Package"]
                        .mean()
                        .round(2)
                        .to_dict()
                    )

                    plt.figure(figsize=(7, 4))

                    sns.boxplot(
                        x="CollegeTier",
                        y="Salary Package",
                        data=placed_data
                    )

                    plt.title(
                        "Salary by College Tier"
                    )

                    _save(
                        "salary_by_tier.png"
                    )

                    charts.append(
                        "charts/salary_by_tier.png"
                    )

                else:
                    salary_by_tier = {}

            else:
                salary_by_tier = {}

        else:
            salary_by_tier = {}

        # 15. PAIRPLOT
        pair_cols = [
            "CGPA",
            "AptitudeTestScore",
            "CodingTestScore",
            "MockInterviewScore",
            "PlacementStatus"
        ]

        pair_cols = [
            c for c in pair_cols
            if c in data.columns
        ]

        if len(pair_cols) >= 3:
            g = sns.pairplot(
                data[pair_cols],
                hue="PlacementStatus",
                diag_kind="hist"
            )

            g.fig.suptitle(
                "Pairwise Relationships",
                y=1.02
            )

            g.savefig(
                _chart_path("pairplot.png"),
                dpi=120,
                bbox_inches="tight"
            )

            plt.close("all")

            charts.append(
                "charts/pairplot.png"
            )

            # RETURN ALL
            return {

                # Dataset Summary
                "rows": rows,
                "columns": columns,
                "duplicates": duplicates,

                # Basic Information
                "basic_info": basic_info.to_dict(orient="index"),
                "dtypes": dtypes,

                # Descriptive Statistics
                "numeric_summary": numeric_summary,
                "categorical_summary": categorical_summary,

                # Missing Values
                "missing": missing_df.to_dict(orient="index"),

                # Target Variable
                "target_counts": target_counts,

                # Correlation
                "correlation_table": correlation_table,

                # Categorical Counts
                "categorical_counts": categorical_counts,

                # Crosstabs
                "gender_placement": (
                    gender_placement.to_dict()
                    if isinstance(gender_placement, pd.DataFrame)
                    else {}
                ),

                "tier_placement": (
                    tier_placement.to_dict()
                    if isinstance(tier_placement, pd.DataFrame)
                    else {}
                ),

                "stream_placement": (
                    stream_placement.to_dict()
                    if isinstance(stream_placement, pd.DataFrame)
                    else {}
                ),

                # SGPA Trend
                "avg_sgpa": avg_sgpa,

                # Salary
                "salary_stats": salary_stats,
                "salary_by_tier": salary_by_tier,

                # All Charts
                "charts": charts
            }

        # --------------------------------------------------------
        # Run Standalone
        # --------------------------------------------------------

        if __name__ == "__main__":

            results = run_eda()

            print("\nEDA Completed Successfully")

            print("\nRows:", results["rows"])

            print("Columns:", results["columns"])

            print("Duplicate Rows:", results["duplicates"])

            print("\nGenerated Charts:")

            for chart in results["charts"]:
                print(chart)