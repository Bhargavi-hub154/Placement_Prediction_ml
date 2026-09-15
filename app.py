from flask import Flask, render_template, request

from load_data import load_data, get_data_summary
from placement_eda import run_eda
from preprocessing import run_preprocessing
from model_training import run_model
from kmeans_clustering import run_kmeans_manual, run_kmeans_elbow, run_kmeans_silhouette


app = Flask(__name__)


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html",
        active="none"
    )


# ============================================================
# DATA LOADING
# ============================================================

@app.route("/data-loading")
def data_loading():

    summary = None
    error = None

    try:

        df = load_data()

        summary = get_data_summary(df)

    except FileNotFoundError as e:

        error = str(e)

    except Exception as e:

        error = f"{type(e).__name__}: {e}"

        import traceback
        traceback.print_exc()

    return render_template(
        "index.html",

        active="data-loading",

        summary=summary,

        error=error
    )


# ============================================================
# EDA
# ============================================================

@app.route("/eda")
def eda_page():

    error = None
    results = None

    try:

        results = run_eda()

    except FileNotFoundError as e:

        error = str(e)

    except Exception as e:

        error = f"{type(e).__name__}: {e}"

        import traceback
        traceback.print_exc()

    return render_template(
        "eda.html",

        active="eda",

        results=results,

        error=error
    )


# ============================================================
# PREPROCESSING
# ============================================================

@app.route("/preprocessing")
def preprocessing_page():

    error = None
    results = None

    try:

        # ----------------------------------------------------
        # ACTION PARAMETERS
        # ----------------------------------------------------

        row_drop = request.args.get(
            "row_drop",
            0,
            type=int
        )

        col_drop = request.args.get(
            "col_drop",
            0,
            type=int
        )

        encode = request.args.get(
            "encode",
            0,
            type=int
        )

        split = request.args.get(
            "split",
            0,
            type=int
        )

        scale = request.args.get(
            "scale",
            None
        )

        # ----------------------------------------------------
        # MISSING VALUE METHOD
        # ----------------------------------------------------

        missing_method = request.args.get(
            "missing_method",
            "median"
        )

        # ----------------------------------------------------
        # VALIDATE SCALE
        # ----------------------------------------------------

        if scale not in (
            None,
            "",
            "standard",
            "minmax"
        ):

            scale = None

        # ----------------------------------------------------
        # ROW/COLUMN ACTION HAS PRIORITY
        # ----------------------------------------------------

        if row_drop:

            missing_method = "row"

        elif col_drop:

            missing_method = "column"

        # ----------------------------------------------------
        # RUN PREPROCESSING
        # ----------------------------------------------------

        results = run_preprocessing(

            do_row_drop=bool(row_drop),

            do_col_drop=bool(col_drop),

            do_encode=bool(encode),

            do_split=bool(split),

            scaling_method=scale,

            missing_method=missing_method
        )

    except FileNotFoundError as e:

        error = str(e)

    except Exception as e:

        error = f"{type(e).__name__}: {e}"

        import traceback
        traceback.print_exc()

    return render_template(
        "preprocessing.html",

        active="preprocessing",

        results=results,

        error=error
    )


# ============================================================
# MODEL TRAINING
# ============================================================

@app.route("/models/<model_key>")
def model_page(model_key):
    try:
        results = run_model(model_key)
        return render_template(
            "models.html",
            active=model_key,
            results=results,
            chart_version=int(__import__("time").time())
        )
    except Exception as e:
        return render_template(
            "index.html",
            active="none",
            error=f"Could not train this model: {type(e).__name__}: {e}"
        )


# ============================================================
# KMEANS CLUSTERING
# ============================================================

@app.route("/kmeans")
def kmeans_page():
    method  = request.args.get("method", None)   # manual | elbow | silhouette
    k_value = request.args.get("k", 3, type=int)

    results = None
    error   = None

    try:
        if method == "manual":
            results = run_kmeans_manual(k=k_value)
        elif method == "elbow":
            results = run_kmeans_elbow()
        elif method == "silhouette":
            results = run_kmeans_silhouette()
        # no method selected yet — just show the selection page
    except Exception as e:
        import traceback
        traceback.print_exc()
        error = f"{type(e).__name__}: {e}"

    return render_template(
        "kmeans.html",
        active="kmeans",
        results=results,
        method=method,
        k_value=k_value,
        error=error,
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)

