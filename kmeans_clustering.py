"""
kmeans_clustering.py
====================
K-Means unsupervised learning — uses placement_Dataset_preprocessed.csv

Charts produced (no PCA):
  • WCSS vs K line plot  — always shown for every method
  • Silhouette score vs K bar chart — shown for silhouette method
  • Cluster-profile bar chart — top discriminating features per cluster
"""

import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED_PATH = os.path.join(BASE_DIR, "placement_Dataset_preprocessed.csv")
RANDOM_STATE      = 42
K_RANGE           = range(2, 11)

DROP_FOR_CLUSTERING = ["PlacementStatus", "Salary Package", "IsAnomaly", "StudentID"]


# ═══════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════

def _load():
    if not os.path.exists(PREPROCESSED_PATH):
        from preprocessing import generate_preprocessed_csv
        generate_preprocessed_csv()

    df = pd.read_csv(PREPROCESSED_PATH)
    df.drop(columns=[c for c in DROP_FOR_CLUSTERING if c in df.columns], inplace=True)
    df_num = df.select_dtypes(include=np.number)
    df_num = df_num.loc[:, df_num.std() > 0]          # drop constant cols
    return df_num.values.astype(float), df_num.columns.tolist()


# ═══════════════════════════════════════════════
# CHART HELPERS
# ═══════════════════════════════════════════════

def _chart_dir():
    d = os.path.join(BASE_DIR, "static", "charts")
    os.makedirs(d, exist_ok=True)
    return d


def _save(fig, name):
    path = os.path.join(_chart_dir(), name)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return f"charts/{name}"


def _wcss_chart(k_vals, wcss_vals, highlight_k=None, title="WCSS vs Number of Clusters (K)"):
    """
    Line plot: x = K, y = WCSS.
    If highlight_k is given, that point is marked with a star and vertical dashed line.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(k_vals, wcss_vals,
            marker="o", color="#ff9a00", lw=2.5,
            ms=8, mfc="#ffffff", mew=2.5, zorder=3)

    # Annotate every WCSS value
    for k, w in zip(k_vals, wcss_vals):
        ax.annotate(
            f"{w:,.0f}",
            xy=(k, w),
            textcoords="offset points",
            xytext=(0, 11),
            ha="center",
            fontsize=8,
            color="#6b3800",
        )

    # Highlight the chosen / elbow K
    if highlight_k and highlight_k in k_vals:
        idx = k_vals.index(highlight_k)
        hw  = wcss_vals[idx]
        ax.scatter([highlight_k], [hw],
                   s=160, color="#b85c00", zorder=5, label=f"Selected K = {highlight_k}")
        ax.axvline(highlight_k, color="#b85c00", lw=1.4, ls="--", alpha=0.55)
        ax.axhline(hw,          color="#b85c00", lw=1.0, ls=":",  alpha=0.40)
        ax.legend(frameon=False, fontsize=10)

    ax.set_xticks(k_vals)
    ax.set_xlabel("Number of Clusters (K)", fontsize=11)
    ax.set_ylabel("WCSS  (Within-Cluster Sum of Squares)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    return _save(fig, "kmeans_wcss.png")


def _sil_chart(k_vals, sil_vals, best_k):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#ff9a00" if k == best_k else "#ffcd4b" for k in k_vals]
    bars = ax.bar(k_vals, sil_vals, color=colors, edgecolor="none", width=0.6)
    for k, s in zip(k_vals, sil_vals):
        ax.text(k, s + 0.003, f"{s:.4f}",
                ha="center", fontsize=8.5, color="#4a3000")
    ax.set_xticks(k_vals)
    ax.set_xlabel("Number of Clusters (K)", fontsize=11)
    ax.set_ylabel("Average Silhouette Score", fontsize=11)
    ax.set_title("Silhouette Score vs K  (higher = better clusters)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # Label best
    idx = k_vals.index(best_k)
    bars[idx].set_edgecolor("#b85c00")
    bars[idx].set_linewidth(2)
    ax.annotate(f"Best K = {best_k}",
                xy=(best_k, sil_vals[idx]),
                xytext=(best_k + 0.5, sil_vals[idx] + 0.012),
                fontsize=9, color="#b85c00", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#b85c00", lw=1.4))

    return _save(fig, "kmeans_silhouette.png")


def _cluster_scatter_chart(X, labels, feat_names, km_model, k):
    """
    2-D scatter: CGPA (x) vs AptitudeTestScore (y) — both already
    StandardScaled in the preprocessed CSV.
    Each dot is coloured by cluster; red ✕ markers show cluster centres.
    Falls back to the first two features if neither column is available.
    """
    cmap = plt.colormaps.get_cmap("tab10").resampled(k)

    # Pick the two axes
    x_feat = "CGPA"            if "CGPA"             in feat_names else feat_names[0]
    y_feat = "AptitudeTestScore" if "AptitudeTestScore" in feat_names else feat_names[min(1, len(feat_names)-1)]

    xi = feat_names.index(x_feat)
    yi = feat_names.index(y_feat)

    # Sub-sample for speed (max 8 000 points plotted)
    rng  = np.random.default_rng(42)
    idx  = rng.choice(len(X), size=min(8000, len(X)), replace=False)
    Xs   = X[idx]
    ls   = labels[idx]

    fig, ax = plt.subplots(figsize=(9, 6))

    for c in range(k):
        mask = ls == c
        ax.scatter(Xs[mask, xi], Xs[mask, yi],
                   s=18, alpha=0.45, color=cmap(c),
                   label=f"Cluster {c}", linewidths=0)

    # Cluster centres
    centres = km_model.cluster_centers_
    ax.scatter(centres[:, xi], centres[:, yi],
               s=220, marker="X", color="red", zorder=5,
               edgecolors="white", linewidths=0.8, label="Cluster Centers")

    ax.set_xlabel(f"{x_feat} (Standardized)", fontsize=11)
    ax.set_ylabel(f"{y_feat} (Standardized)", fontsize=11)
    ax.set_title(f"K-Means Clustering  (K = {k})", fontsize=13, fontweight="bold")
    ax.legend(frameon=True, fontsize=9, markerscale=1.5)
    ax.grid(alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)

    return _save(fig, f"kmeans_scatter_k{k}.png")


def _profile_chart(X, labels, feat_names, k):
    """
    Grouped bar chart: top-10 most discriminating features, one group per feature,
    one bar per cluster. No PCA involved.
    """
    means    = np.array([X[labels == c].mean(axis=0) for c in range(k)])  # k × F
    variance = means.var(axis=0)
    top_idx  = np.argsort(variance)[::-1][:10]
    top_feat = [feat_names[i] for i in top_idx]
    top_m    = means[:, top_idx]

    # Normalise to [0, 1] per feature
    col_min = top_m.min(axis=0)
    col_max = top_m.max(axis=0)
    rng     = np.where(col_max - col_min == 0, 1.0, col_max - col_min)
    top_n   = (top_m - col_min) / rng

    cmap  = plt.colormaps.get_cmap("tab10").resampled(k)
    x     = np.arange(len(top_feat))
    w     = 0.8 / k

    fig, ax = plt.subplots(figsize=(13, max(4.5, k * 1.3)))
    for i in range(k):
        offset = (i - k / 2) * w + w / 2
        ax.bar(x + offset, top_n[i], width=w * 0.9,
               color=cmap(i), alpha=0.88, label=f"Cluster {i}")

    ax.set_xticks(x)
    ax.set_xticklabels(top_feat, rotation=38, ha="right", fontsize=9)
    ax.set_ylabel("Normalised mean  (0 = lowest cluster, 1 = highest)")
    ax.set_title(f"Cluster Profiles — Top 10 Discriminating Features  (K = {k})",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)

    return _save(fig, f"kmeans_profile_k{k}.png")


# ═══════════════════════════════════════════════
# CORE RUNNER
# ═══════════════════════════════════════════════

def _run(X, feat_names, k, wcss_k_vals, wcss_vals, highlight_k=None, wcss_title="WCSS vs K"):
    """
    Fit KMeans(k), generate WCSS chart + cluster scatter + profile chart.
    wcss_k_vals / wcss_vals are passed in (already computed).
    """
    km     = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
    labels = km.fit_predict(X)

    # Silhouette
    sample = min(5000, len(X))
    try:
        sil = round(float(silhouette_score(
            X, labels, sample_size=sample, random_state=RANDOM_STATE
        )), 4)
    except Exception:
        sil = None

    wcss  = round(float(km.inertia_), 2)
    sizes = {f"Cluster {c}": int((labels == c).sum()) for c in range(k)}

    # Per-cluster feature means
    means_list = []
    for c in range(k):
        row = {"Cluster": c}
        row.update({
            f: round(float(X[labels == c, i].mean()), 4)
            for i, f in enumerate(feat_names)
        })
        means_list.append(row)

    chart_wcss    = _wcss_chart(wcss_k_vals, wcss_vals,
                                highlight_k=highlight_k or k,
                                title=wcss_title)
    chart_scatter = _cluster_scatter_chart(X, labels, feat_names, km, k)
    chart_profile = _profile_chart(X, labels, feat_names, k)

    return {
        "k":             k,
        "wcss":          wcss,
        "silhouette":    sil,
        "cluster_sizes": sizes,
        "cluster_means": means_list,
        "feat_names":    feat_names,
        "chart_wcss":    chart_wcss,
        "chart_scatter": chart_scatter,
        "chart_profile": chart_profile,
        "n_samples":     len(X),
        "n_features":    len(feat_names),
        "wcss_table":    [{"k": kv, "wcss": wv}
                          for kv, wv in zip(wcss_k_vals, wcss_vals)],
    }


# ═══════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════

def run_kmeans_manual(k: int):
    """
    Manual K — user picks the number of clusters.
    Still computes WCSS for K=2..10 so the chart always shows context.
    The chosen K is highlighted on the plot.
    """
    if k < 2:
        return {"error": "K must be ≥ 2."}

    X, feat_names = _load()

    # Compute WCSS for full range so chart has context
    k_vals, wcss_vals = [], []
    for kv in K_RANGE:
        km = KMeans(n_clusters=kv, n_init=20, random_state=RANDOM_STATE)
        km.fit(X)
        k_vals.append(kv)
        wcss_vals.append(round(float(km.inertia_), 2))

    # Extend range if user chose K outside K_RANGE
    if k not in k_vals:
        km_extra = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        km_extra.fit(X)
        k_vals.append(k)
        wcss_vals.append(round(float(km_extra.inertia_), 2))
        k_vals_sorted = sorted(range(len(k_vals)), key=lambda i: k_vals[i])
        k_vals   = [k_vals[i]   for i in k_vals_sorted]
        wcss_vals = [wcss_vals[i] for i in k_vals_sorted]

    result = _run(
        X, feat_names, k,
        wcss_k_vals=k_vals,
        wcss_vals=wcss_vals,
        highlight_k=k,
        wcss_title=f"WCSS vs K  (selected K = {k} highlighted)",
    )
    result.update({
        "method":      "Manual K Selection",
        "method_icon": "✏️",
        "method_desc": (
            f"K = {k} was set manually. "
            "The WCSS chart shows all K from 2–10 for context; "
            "your chosen K is marked with a filled dot and dashed lines."
        ),
        "data_source": "placement_Dataset_preprocessed.csv",
    })
    return result


def run_kmeans_elbow():
    """
    Elbow Method — compute WCSS for K=2..10, pick the elbow automatically.
    """
    X, feat_names = _load()

    k_vals, wcss_vals = [], []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        km.fit(X)
        k_vals.append(k)
        wcss_vals.append(round(float(km.inertia_), 2))

    # Elbow = largest drop in WCSS (first derivative maximum)
    diffs  = [wcss_vals[i] - wcss_vals[i+1] for i in range(len(wcss_vals) - 1)]
    best_k = k_vals[diffs.index(max(diffs)) + 1]

    result = _run(
        X, feat_names, best_k,
        wcss_k_vals=k_vals,
        wcss_vals=wcss_vals,
        highlight_k=best_k,
        wcss_title=f"Elbow Method — WCSS vs K  (elbow at K = {best_k})",
    )
    result.update({
        "method":      "Elbow Method",
        "method_icon": "📐",
        "method_desc": (
            "WCSS is plotted for K = 2 to 10. "
            "The elbow — where the curve bends and further K additions give "
            "diminishing returns — indicates the best K. "
            f"Automatically detected elbow: K = {best_k}."
        ),
        "best_k":      best_k,
        "data_source": "placement_Dataset_preprocessed.csv",
    })
    return result


def run_kmeans_silhouette():
    """
    Silhouette Method — pick K that maximises avg silhouette score.
    Also always shows the WCSS vs K chart alongside the silhouette chart.
    """
    X, feat_names = _load()

    k_vals, wcss_vals, sil_vals = [], [], []
    sample = min(5000, len(X))

    for k in K_RANGE:
        km     = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = km.fit_predict(X)
        k_vals.append(k)
        wcss_vals.append(round(float(km.inertia_), 2))
        sil_vals.append(round(float(
            silhouette_score(X, labels, sample_size=sample, random_state=RANDOM_STATE)
        ), 4))

    best_k    = k_vals[sil_vals.index(max(sil_vals))]
    chart_sil = _sil_chart(k_vals, sil_vals, best_k)

    result = _run(
        X, feat_names, best_k,
        wcss_k_vals=k_vals,
        wcss_vals=wcss_vals,
        highlight_k=best_k,
        wcss_title=f"WCSS vs K  (best silhouette at K = {best_k})",
    )
    result.update({
        "method":      "Silhouette Method",
        "method_icon": "📏",
        "method_desc": (
            "Average silhouette score s(i) = (b(i)−a(i)) / max(a(i),b(i)) "
            "is computed for K = 2 to 10. "
            "Score near +1 → dense, well-separated clusters. "
            f"Best K found: {best_k}."
        ),
        "best_k":      best_k,
        "chart_sil":   chart_sil,
        "sil_table":   [{"k": kv, "sil": sv}
                        for kv, sv in zip(k_vals, sil_vals)],
        "data_source": "placement_Dataset_preprocessed.csv",
    })
    return result
