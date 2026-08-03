import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from load_data import load_data

# Folder where charts will be saved
CHARTS_DIR = os.path.join(os.path.dirname(__file__), "static", "charts")


def _chart_path(filename):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return os.path.join(CHARTS_DIR, filename)


def _save(filename):
    plt.tight_layout()
    plt.savefig(_chart_path(filename), dpi=300, bbox_inches="tight")
    plt.close()


def run_eda():
    # Load dataset
    data = load_data()

    charts = []

    sns.set_style("whitegrid")
    sns.set_context("paper")

    # ======================================
    # Missing Values
    # ======================================
    missing = data.isnull().sum()
    missing_pct = (missing / len(data)) * 100

    missing_df = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": missing_pct
    })

    missing_df = missing_df[missing_df["missing_count"] > 0]

    print("\nMissing Values")
    print(missing_df)

    if not missing_df.empty:

        plt.figure(figsize=(10, 5))

        sns.barplot(
            x=missing_df.index,
            y=missing_df["missing_pct"]
        )

        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Missing %")
        plt.title("Missing Values by Column")

        _save("missing_values.png")
        charts.append("charts/missing_values.png")

        plt.figure(figsize=(12, 6))

        sns.heatmap(
            data.isnull(),
            cbar=False,
            cmap="viridis"
        )

        plt.title("Missing Values Heatmap")

        _save("missing_heatmap.png")
        charts.append("charts/missing_heatmap.png")

    # ======================================
    # Duplicate Rows
    # ======================================
    duplicates = int(data.duplicated().sum())

    print("\nDuplicate Rows:", duplicates)

    # ======================================
    # Placement Status Distribution
    # ======================================
    print("\nPlacement Status")
    print(data["PlacementStatus"].value_counts())

    target_counts = data["PlacementStatus"].value_counts().to_dict()

    plt.figure(figsize=(6, 5))

    sns.countplot(
        x="PlacementStatus",
        data=data
    )

    plt.title("Placement Status Distribution")
    plt.xlabel("Placement Status (0 = Not Placed, 1 = Placed)")
    plt.ylabel("Count")

    _save("placement_status.png")
    charts.append("charts/placement_status.png")

    # ======================================
    # Return Results
    # ======================================
    return {
        "rows": len(data),
        "columns": len(data.columns),
        "duplicates": duplicates,
        "missing": missing_df.to_dict(orient="index"),
        "target_counts": target_counts,
        "charts": charts
    }


if __name__ == "__main__":
    results = run_eda()
    print(results)