import os

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder
)


# ============================================================
# DATASET PATH
# ============================================================

FILE_PATH = (
    r"D:\sem4\mlProject\placement_Dataset.csv"
)


# ============================================================
# HELPER - NUMERIC CLEANING
# ============================================================

def to_numeric_clean(series):

    if pd.api.types.is_numeric_dtype(series):

        return pd.to_numeric(
            series,
            errors="coerce"
        )

    return pd.to_numeric(

        series
        .astype(str)
        .str.replace(
            ",",
            "",
            regex=False
        )
        .str.replace(
            "₹",
            "",
            regex=False
        )
        .str.replace(
            "$",
            "",
            regex=False
        )
        .str.strip(),

        errors="coerce"
    )


# ============================================================
# STANDARDIZE
# ============================================================

def standardize(x):

    x = np.asarray(
        x,
        dtype=float
    )

    mean = x.mean()

    std = x.std()

    if (
        std == 0
        or not np.isfinite(std)
    ):

        std = 1.0

    return (
        (x - mean) / std,
        mean,
        std
    )


# ============================================================
# BATCH GRADIENT DESCENT
# ============================================================

def batch_gradient_descent(
    x,
    y,
    alpha=0.05,
    epochs=6
):

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    m = len(x)

    if m == 0:

        return (
            0.0,
            0.0,
            []
        )

    theta0 = 0.0

    theta1 = 0.0

    mse_history = []

    for epoch in range(
        1,
        epochs + 1
    ):

        # ----------------------------------------------------
        # CURRENT PREDICTION
        # ----------------------------------------------------

        y_hat = (
            theta0
            + theta1 * x
        )

        # ----------------------------------------------------
        # ERROR
        # ----------------------------------------------------

        error = (
            y_hat - y
        )

        # ----------------------------------------------------
        # GRADIENTS
        # ----------------------------------------------------

        grad_theta0 = (
            (2.0 / m)
            * np.sum(error)
        )

        grad_theta1 = (
            (2.0 / m)
            * np.sum(
                error * x
            )
        )

        # ----------------------------------------------------
        # UPDATE
        # ----------------------------------------------------

        theta0 = (
            theta0
            - alpha * grad_theta0
        )

        theta1 = (
            theta1
            - alpha * grad_theta1
        )

        # ----------------------------------------------------
        # MSE AFTER UPDATE
        # ----------------------------------------------------

        updated_y_hat = (
            theta0
            + theta1 * x
        )

        mse = np.mean(
            (
                updated_y_hat - y
            ) ** 2
        )

        mse_history.append({

            "epoch":
                int(epoch),

            "theta0":
                float(theta0),

            "theta1":
                float(theta1),

            "mse":
                float(mse)
        })

    return (
        theta0,
        theta1,
        mse_history
    )


# ============================================================
# CHART DIRECTORY
# ============================================================

def _charts_directory():

    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    chart_dir = os.path.join(
        base_dir,
        "static",
        "charts"
    )

    os.makedirs(
        chart_dir,
        exist_ok=True
    )

    return chart_dir


# ============================================================
# CLEAR OLD CHARTS
# ============================================================

def _clear_old_regression_charts(
    chart_dir
):

    chart_names = [

        "gradient_descent_mse.png",

        "gradient_descent_vs_sklearn.png",

        "scaling_comparison.png",

        "lr_cgpa_salary.png",

        "lr_gradient_descent_mse.png",

        "lr_model_comparison.png"
    ]

    for filename in chart_names:

        path = os.path.join(
            chart_dir,
            filename
        )

        if os.path.exists(path):

            try:

                os.remove(path)

            except OSError:

                pass


# ============================================================
# SAVE CHART
# ============================================================

def _save_figure(
    fig,
    filename
):

    chart_dir = (
        _charts_directory()
    )

    path = os.path.join(
        chart_dir,
        filename
    )

    fig.tight_layout()

    fig.savefig(
        path,
        format="png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    return filename


# ============================================================
# GENERATE REGRESSION CHARTS
# ============================================================

def generate_regression_charts(
    model_data
):

    chart_dir = (
        _charts_directory()
    )

    _clear_old_regression_charts(
        chart_dir
    )

    charts = []

    if not model_data:

        return charts

    if model_data.get("error"):

        return charts

    # ========================================================
    # DATA
    # ========================================================

    x_train = np.asarray(
        model_data["x_train"],
        dtype=float
    )

    x_val = np.asarray(
        model_data["x_val"],
        dtype=float
    )

    y_val = np.asarray(
        model_data["y_val"],
        dtype=float
    )

    gd_predictions = np.asarray(
        model_data["gd_predictions"],
        dtype=float
    )

    sk_predictions = np.asarray(
        model_data["sk_predictions"],
        dtype=float
    )

    mse_history = (
        model_data["mse_history"]
    )

    # ========================================================
    # CHART 1 - MSE
    # ========================================================

    epochs = [

        item["epoch"]

        for item in mse_history
    ]

    mse_values = [

        item["mse"]

        for item in mse_history
    ]

    fig, ax = plt.subplots(
        figsize=(8.5, 5.2)
    )

    if epochs:

        ax.plot(

            epochs,

            mse_values,

            marker="o",

            linewidth=2,

            color="#ff9800"
        )

        for epoch, mse in zip(
            epochs,
            mse_values
        ):

            ax.annotate(

                f"{mse:.2f}",

                (epoch, mse),

                textcoords="offset points",

                xytext=(0, 8),

                ha="center",

                fontsize=9
            )

        ax.set_xticks(
            epochs
        )

    ax.set_title(
        "Gradient Descent MSE - 6 Iterations"
    )

    ax.set_xlabel(
        "Iteration"
    )

    ax.set_ylabel(
        "Mean Squared Error"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    filename = (
        "gradient_descent_mse.png"
    )

    _save_figure(
        fig,
        filename
    )

    charts.append({

        "title":
            "Gradient Descent MSE",

        "filename":
            filename
    })

    # ========================================================
    # CHART 2 - GD VS SKLEARN
    # ========================================================

    sort_index = np.argsort(
        x_val
    )

    sorted_x = (
        x_val[
            sort_index
        ]
    )

    sorted_actual = (
        y_val[
            sort_index
        ]
    )

    sorted_gd = (
        gd_predictions[
            sort_index
        ]
    )

    sorted_sk = (
        sk_predictions[
            sort_index
        ]
    )

    fig, ax = plt.subplots(
        figsize=(8.5, 5.2)
    )

    ax.scatter(

        sorted_x,

        sorted_actual,

        s=18,

        alpha=0.35,

        label="Actual Salary",

        color="#777777"
    )

    ax.plot(

        sorted_x,

        sorted_gd,

        linewidth=2,

        label="Gradient Descent",

        color="#ff9800"
    )

    ax.plot(

        sorted_x,

        sorted_sk,

        linewidth=2,

        linestyle="--",

        label="Scikit-learn",

        color="#2e7d32"
    )

    ax.set_title(
        "Gradient Descent vs Scikit-learn"
    )

    ax.set_xlabel(
        "CGPA"
    )

    ax.set_ylabel(
        "Salary Package"
    )

    ax.legend()

    ax.grid(
        True,
        alpha=0.25
    )

    filename = (
        "gradient_descent_vs_sklearn.png"
    )

    _save_figure(
        fig,
        filename
    )

    charts.append({

        "title":
            "Gradient Descent vs Scikit-learn",

        "filename":
            filename
    })

    # ========================================================
    # CHART 3 - SCALING
    # ========================================================

    x_mean = float(
        model_data["x_mean"]
    )

    x_std = float(
        model_data["x_std"]
    )

    gd_theta0 = float(
        model_data["gd_theta0"]
    )

    gd_theta1 = float(
        model_data["gd_theta1"]
    )

    if x_std == 0:

        x_std = 1.0

    x_original_line = np.linspace(

        float(
            x_train.min()
        ),

        float(
            x_train.max()
        ),

        200
    )

    x_scaled_line = (
        x_original_line - x_mean
    ) / x_std

    y_scaled_prediction = (

        gd_theta0

        + gd_theta1
        * x_scaled_line
    )

    original_slope = (
        gd_theta1 / x_std
    )

    original_intercept = (

        gd_theta0

        - (
            gd_theta1
            * x_mean
            / x_std
        )
    )

    y_original_prediction = (

        original_intercept

        + original_slope
        * x_original_line
    )

    fig, ax = plt.subplots(
        figsize=(8.5, 5.2)
    )

    ax.plot(

        x_original_line,

        y_original_prediction,

        linewidth=2,

        label="Original CGPA Scale",

        color="#ff9800"
    )

    ax.plot(

        x_original_line,

        y_scaled_prediction,

        linewidth=2,

        linestyle="--",

        label="Standardized CGPA Scale",

        color="#1976d2"
    )

    ax.scatter(

        x_train,

        np.asarray(
            model_data["y_train"],
            dtype=float
        ),

        s=15,

        alpha=0.25,

        label="Training Data"
    )

    ax.set_title(
        "Linear Regression - Scaling Comparison"
    )

    ax.set_xlabel(
        "CGPA"
    )

    ax.set_ylabel(
        "Salary Package"
    )

    ax.legend()

    ax.grid(
        True,
        alpha=0.25
    )

    filename = (
        "scaling_comparison.png"
    )

    _save_figure(
        fig,
        filename
    )

    charts.append({

        "title":
            "Scaling Comparison",

        "filename":
            filename
    })

    return charts


# ============================================================
# MAIN PREPROCESSING
# ============================================================

def run_preprocessing(

    do_row_drop=False,

    do_col_drop=False,

    do_encode=False,

    do_split=False,

    scaling_method=None,

    missing_method="median"
):

    # ========================================================
    # LOAD DATA
    # ========================================================

    df = pd.read_csv(
        FILE_PATH
    )

    results = {}

    original_rows = len(df)

    original_columns = len(
        df.columns
    )

    results["n_rows"] = int(
        original_rows
    )

    results["n_columns"] = int(
        original_columns
    )

    # ========================================================
    # DETERMINE MISSING METHOD
    # ========================================================

    if do_row_drop:

        missing_method = "row"

    elif do_col_drop:

        missing_method = "column"

    valid_methods = {

        "row",

        "column",

        "mean",

        "median"
    }

    if missing_method not in valid_methods:

        missing_method = "median"

    # ========================================================
    # MISSING BEFORE
    # ========================================================

    missing_before_columns = (
        df.isnull().sum()
    )

    missing_before = int(
        missing_before_columns.sum()
    )

    changes = {

        "rows_deleted": [],

        "columns_deleted": [],

        "values_imputed": []
    }

    # ========================================================
    # ROW DELETION
    # ========================================================

    if missing_method == "row":

        rows_with_missing = (

            df.index[
                df.isnull().any(
                    axis=1
                )
            ].tolist()
        )

        changes[
            "rows_deleted"
        ] = [

            int(index)

            for index
            in rows_with_missing
        ]

        df = (
            df
            .dropna(
                axis=0
            )
            .copy()
        )

    # ========================================================
    # COLUMN DELETION
    # ========================================================

    elif missing_method == "column":

        # Delete columns with 50% or more missing values.
        threshold = len(df) * 0.50

        missing_columns = (

            df.columns[
                df.isnull().sum()
                >= threshold
            ].tolist()
        )

        changes[
            "columns_deleted"
        ] = list(
            missing_columns
        )

        df = (
            df
            .drop(
                columns=missing_columns
            )
            .copy()
        )

    # ========================================================
    # MEAN / MEDIAN IMPUTATION
    # ========================================================

    else:

        numerical_columns = (

            df.select_dtypes(
                include=np.number
            )
            .columns
            .tolist()
        )

        for col in numerical_columns:

            count = int(
                df[col]
                .isnull()
                .sum()
            )

            if count <= 0:

                continue

            if missing_method == "mean":

                value = (
                    df[col].mean()
                )

                method_name = "Mean"

            else:

                value = (
                    df[col].median()
                )

                method_name = "Median"

            if pd.notna(value):

                df[col] = (
                    df[col]
                    .fillna(value)
                )

                changes[
                    "values_imputed"
                ].append({

                    "column":
                        col,

                    "count":
                        count,

                    "method":
                        method_name,

                    "value":
                        round(
                            float(value),
                            4
                        )
                })

        categorical_columns = (

            df.select_dtypes(
                exclude=np.number
            )
            .columns
            .tolist()
        )

        for col in categorical_columns:

            count = int(
                df[col]
                .isnull()
                .sum()
            )

            if count <= 0:

                continue

            mode_values = (
                df[col]
                .mode()
            )

            if len(mode_values) > 0:

                value = (
                    mode_values.iloc[0]
                )

                df[col] = (
                    df[col]
                    .fillna(value)
                )

                changes[
                    "values_imputed"
                ].append({

                    "column":
                        col,

                    "count":
                        count,

                    "method":
                        "Mode",

                    "value":
                        str(value)
                })

    # ========================================================
    # MISSING AFTER
    # ========================================================

    missing_after_columns = (
        df.isnull().sum()
    )

    missing_after = int(
        missing_after_columns.sum()
    )

    method_names = {

        "row":
            "Row-wise Deletion",

        "column":
            "Column-wise Deletion",

        "mean":
            "Mean Imputation",

        "median":
            "Median Imputation"
    }

    results[
        "missing_values"
    ] = {

        "method":
            missing_method,

        "method_name":
            method_names[
                missing_method
            ],

        "missing_before":
            missing_before,

        "missing_after":
            missing_after,

        "rows_before":
            original_rows,

        "rows_after":
            int(len(df)),

        "columns_before":
            original_columns,

        "columns_after":
            int(len(df.columns)),

        "rows_deleted":
            original_rows - len(df),

        "columns_deleted":
            original_columns
            - len(df.columns),

        "rows_with_missing":
            int(
                missing_before_columns
                .gt(0)
                .sum()
            )
    }

    # ========================================================
    # MISSING TABLE
    # ========================================================

    missing_table = []

    for col, count in (

        missing_before_columns[
            missing_before_columns > 0
        ].items()
    ):

        missing_table.append({

            "column":
                col,

            "before":
                int(count),

            "after":
                (
                    int(
                        missing_after_columns
                        .get(
                            col,
                            0
                        )
                    )

                    if col in df.columns

                    else "Deleted"
                ),

            "percentage":
                round(
                    (
                        count
                        / original_rows
                        * 100
                    ),
                    2
                )
        })

    results[
        "missing_table"
    ] = missing_table

    results[
        "changes"
    ] = changes

    # ========================================================
    # IQR OUTLIER DETECTION
    # ========================================================

    IQR_COL = "CodingTestScore"

    if IQR_COL in df.columns:

        series = (
            to_numeric_clean(
                df[IQR_COL]
            )
            .dropna()
        )

        if len(series) > 0:

            Q1 = series.quantile(
                0.25
            )

            Q3 = series.quantile(
                0.75
            )

            IQR = Q3 - Q1

            lower_bound = (
                Q1
                - 1.5 * IQR
            )

            upper_bound = (
                Q3
                + 1.5 * IQR
            )

            outlier_mask = (

                (series < lower_bound)

                |

                (series > upper_bound)
            )

            df[
                "CodingTestScore_clipped"
            ] = (

                to_numeric_clean(
                    df[IQR_COL]
                )
                .clip(
                    lower=lower_bound,
                    upper=upper_bound
                )
            )

            results["iqr"] = {

                "column":
                    IQR_COL,

                "q1":
                    round(
                        float(Q1),
                        4
                    ),

                "q3":
                    round(
                        float(Q3),
                        4
                    ),

                "iqr":
                    round(
                        float(IQR),
                        4
                    ),

                "lower_bound":
                    round(
                        float(
                            lower_bound
                        ),
                        4
                    ),

                "upper_bound":
                    round(
                        float(
                            upper_bound
                        ),
                        4
                    ),

                "n_outliers":
                    int(
                        outlier_mask.sum()
                    ),

                "min_before":
                    round(
                        float(
                            series.min()
                        ),
                        4
                    ),

                "max_before":
                    round(
                        float(
                            series.max()
                        ),
                        4
                    ),

                "min_after":
                    round(
                        float(
                            df[
                                "CodingTestScore_clipped"
                            ].min()
                        ),
                        4
                    ),

                "max_after":
                    round(
                        float(
                            df[
                                "CodingTestScore_clipped"
                            ].max()
                        ),
                        4
                    )
            }

        else:

            results["iqr"] = {}

    else:

        results["iqr"] = {}

    # ========================================================
    # NUMERIC COLUMNS
    # ========================================================

    requested_num_cols = [

        "StudentID",

        "SGPA_Sem1",
        "SGPA_Sem2",
        "SGPA_Sem3",
        "SGPA_Sem4",
        "SGPA_Sem5",
        "SGPA_Sem6",
        "SGPA_Sem7",
        "SGPA_Sem8",

        "CGPA",

        "AttendancePercent",

        "Internships",
        "Projects",
        "Workshops",
        "Certifications",
        "Publications",

        "AptitudeTestScore",

        "SoftSkillsRating",

        "CodingTestScore",

        "CodingTestScore_clipped",

        "MockInterviewScore",

        "ExtraCurricular",

        "IsAnomaly",

        "Salary Package"
    ]

    num_cols = [

        col

        for col in requested_num_cols

        if col in df.columns
    ]

    for col in num_cols:

        df[col] = to_numeric_clean(
            df[col]
        )

    num_cols = [

        col

        for col in num_cols

        if df[col].notna().any()
    ]

    results[
        "numeric_cols"
    ] = num_cols

    # ========================================================
    # CATEGORICAL COLUMNS
    # ========================================================

    requested_categorical_cols = [

        "Gender",
        "City",
        "Stream",
        "Specialisation",
        "Hostel",
        "HistoryOfBacklogs",
        "ExtraCurricular"
    ]

    categorical_cols = [

        col

        for col
        in requested_categorical_cols

        if col in df.columns
    ]

    results[
        "categorical_cols"
    ] = categorical_cols

    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    TARGET = "PlacementStatus"

    if len(df) >= 2:

        if (
            TARGET in df.columns
            and df[TARGET]
            .notna()
            .all()
        ):

            try:

                train_df, test_df = (
                    train_test_split(

                        df,

                        test_size=0.30,

                        random_state=42,

                        stratify=df[
                            TARGET
                        ]
                    )
                )

                split_stratified = True

            except ValueError:

                train_df, test_df = (
                    train_test_split(

                        df,

                        test_size=0.30,

                        random_state=42
                    )
                )

                split_stratified = False

        else:

            train_df, test_df = (
                train_test_split(

                    df,

                    test_size=0.30,

                    random_state=42
                )
            )

            split_stratified = False

    else:

        train_df = df.copy()

        test_df = df.copy()

        split_stratified = False

    results["split"] = {

        "total":
            int(len(df)),

        "train_rows":
            int(len(train_df)),

        "test_rows":
            int(len(test_df)),

        "train_pct":
            round(
                len(train_df)
                / max(len(df), 1)
                * 100,
                1
            ),

        "test_pct":
            round(
                len(test_df)
                / max(len(df), 1)
                * 100,
                1
            ),

        "stratified":
            split_stratified
    }

    # ========================================================
    # TARGET DISTRIBUTION
    # ========================================================

    if TARGET in train_df.columns:

        y_train_counts = (

            train_df[
                TARGET
            ]
            .value_counts()
            .to_dict()
        )

        y_test_counts = (

            test_df[
                TARGET
            ]
            .value_counts()
            .to_dict()
        )

    else:

        y_train_counts = {}

        y_test_counts = {}

    results[
        "split"
    ][
        "y_train_counts"
    ] = {

        str(k): int(v)

        for k, v
        in y_train_counts.items()
    }

    results[
        "split"
    ][
        "y_test_counts"
    ] = {

        str(k): int(v)

        for k, v
        in y_test_counts.items()
    }

    # ========================================================
    # ONE HOT ENCODING
    # ========================================================

    onehot_columns = []

    train_onehot_df = pd.DataFrame(
        index=train_df.index
    )

    if categorical_cols:

        try:

            onehot_encoder = (
                OneHotEncoder(

                    handle_unknown="ignore",

                    sparse_output=False
                )
            )

        except TypeError:

            onehot_encoder = (
                OneHotEncoder(

                    handle_unknown="ignore",

                    sparse=False
                )
            )

        train_onehot_array = (

            onehot_encoder
            .fit_transform(

                train_df[
                    categorical_cols
                ]
                .fillna("Missing")
                .astype(str)
            )
        )

        onehot_columns = (

            onehot_encoder
            .get_feature_names_out(
                categorical_cols
            )
            .tolist()
        )

        train_onehot_df = (
            pd.DataFrame(

                train_onehot_array,

                columns=onehot_columns,

                index=train_df.index
            )
        )

    results[
        "onehot_columns"
    ] = onehot_columns

    results[
        "onehot_preview"
    ] = (

        train_onehot_df
        .head(5)
        .round(4)
        .to_dict(
            orient="records"
        )
    )

    # ========================================================
    # ORDINAL ENCODING
    # ========================================================

    ordinal_cols = [

        col

        for col in [
            "CollegeTier",
            "CGPA_Tier"
        ]

        if col in df.columns
    ]

    ordinal_categories = []

    for col in ordinal_cols:

        if col == "CollegeTier":

            ordinal_categories.append([

                "Tier3",

                "Tier2",

                "Tier1",

                "Missing"
            ])

        elif col == "CGPA_Tier":

            ordinal_categories.append([

                "Low",

                "Mid",

                "High",

                "Missing"
            ])

    train_ordinal_df = pd.DataFrame(
        index=train_df.index
    )

    if ordinal_cols:

        ordinal_encoder = (
            OrdinalEncoder(

                categories=
                    ordinal_categories,

                handle_unknown=
                    "use_encoded_value",

                unknown_value=-1
            )
        )

        train_ordinal_array = (

            ordinal_encoder
            .fit_transform(

                train_df[
                    ordinal_cols
                ]
                .fillna("Missing")
                .astype(str)
            )
        )

        train_ordinal_df = (
            pd.DataFrame(

                train_ordinal_array,

                columns=ordinal_cols,

                index=train_df.index
            )
        )

    results[
        "ordinal_columns"
    ] = ordinal_cols

    results[
        "ordinal_preview"
    ] = (

        train_ordinal_df
        .head(5)
        .round(4)
        .to_dict(
            orient="records"
        )
    )

    # ========================================================
    # TARGET ENCODING
    # ========================================================

    target_encoding_train = (
        pd.DataFrame(
            index=train_df.index
        )
    )

    if TARGET in train_df.columns:

        target_values = (
            train_df[TARGET]
        )

        target_mapping = {

            "Placed": 1,

            "Not Placed": 0,

            "NotPlaced": 0,

            "Yes": 1,

            "No": 0,

            "True": 1,

            "False": 0
        }

        train_target_numeric = (
            target_values.map(
                target_mapping
            )
        )

        numeric_target = (
            to_numeric_clean(
                target_values
            )
        )

        train_target_numeric = (
            train_target_numeric
            .fillna(
                numeric_target
            )
        )

        if train_target_numeric.isna().any():

            unique_values = (

                train_df[TARGET]
                .dropna()
                .unique()
                .tolist()
            )

            target_map = {

                value:
                    index

                for index, value
                in enumerate(
                    unique_values
                )
            }

            train_target_numeric = (
                train_df[TARGET]
                .map(target_map)
            )

        global_mean = float(
            train_target_numeric.mean()
        )

        for col in categorical_cols:

            mapping = (

                pd.DataFrame({

                    "category":
                        train_df[col]
                        .fillna(
                            "Missing"
                        )
                        .astype(str),

                    "target":
                        train_target_numeric

                })

                .groupby(
                    "category"
                )[
                    "target"
                ]

                .mean()
            )

            target_encoding_train[
                col
                + "_TargetEncoded"
            ] = (

                train_df[col]
                .fillna(
                    "Missing"
                )
                .astype(str)
                .map(mapping)
                .fillna(
                    global_mean
                )
            )

        results[
            "global_target_mean"
        ] = round(
            global_mean,
            4
        )

    else:

        results[
            "global_target_mean"
        ] = 0.0

    results[
        "target_columns"
    ] = list(
        target_encoding_train.columns
    )

    results[
        "target_preview"
    ] = (

        target_encoding_train
        .head(5)
        .round(4)
        .to_dict(
            orient="records"
        )
    )

    # ========================================================
    # EMBEDDING IDS
    # ========================================================

    embedding_train = (
        pd.DataFrame(
            index=train_df.index
        )
    )

    for col in categorical_cols:

        categories = (

            train_df[col]
            .fillna(
                "Missing"
            )
            .astype(str)
            .unique()
            .tolist()
        )

        category_to_id = {

            category:
                index + 1

            for index, category
            in enumerate(
                categories
            )
        }

        embedding_train[
            col
            + "_EmbeddingID"
        ] = (

            train_df[col]
            .fillna(
                "Missing"
            )
            .astype(str)
            .map(
                category_to_id
            )
            .fillna(0)
            .astype(int)
        )

    results[
        "embedding_columns"
    ] = list(
        embedding_train.columns
    )

    results[
        "embedding_preview"
    ] = (

        embedding_train
        .head(5)
        .to_dict(
            orient="records"
        )
    )

    # ========================================================
    # SCALING
    # ========================================================

    usable_scale_cols = [

        col

        for col in num_cols

        if col in train_df.columns
        and train_df[col].notna().all()
    ]

    train_before_scaling = (
        train_df[
            usable_scale_cols
        ].copy()
    )

    results[
        "before_scaling"
    ] = (

        train_before_scaling
        .head(5)
        .round(4)
        .to_dict(
            orient="records"
        )
    )

    train_minmax_df = pd.DataFrame(
        index=train_df.index
    )

    train_standard_df = pd.DataFrame(
        index=train_df.index
    )

    if usable_scale_cols:

        minmax_scaler = (
            MinMaxScaler()
        )

        train_minmax = (

            minmax_scaler
            .fit_transform(
                train_before_scaling
            )
        )

        train_minmax_df = (
            pd.DataFrame(

                train_minmax,

                columns=
                    usable_scale_cols,

                index=
                    train_df.index
            )
        )

        standard_scaler = (
            StandardScaler()
        )

        train_standard = (

            standard_scaler
            .fit_transform(
                train_before_scaling
            )
        )

        train_standard_df = (
            pd.DataFrame(

                train_standard,

                columns=
                    usable_scale_cols,

                index=
                    train_df.index
            )
        )

    results[
        "minmax_preview"
    ] = (

        train_minmax_df
        .head(5)
        .round(4)
        .to_dict(
            orient="records"
        )
    )

    results[
        "standard_preview"
    ] = (

        train_standard_df
        .head(5)
        .round(4)
        .to_dict(
            orient="records"
        )
    )

    # ========================================================
    # ACTIVE SCALING METHOD
    # ========================================================

    results[
        "active_scaling"
    ] = scaling_method

    if scaling_method == "standard":

        results[
            "active_scaling_preview"
        ] = results[
            "standard_preview"
        ]

    elif scaling_method == "minmax":

        results[
            "active_scaling_preview"
        ] = results[
            "minmax_preview"
        ]

    else:

        results[
            "active_scaling_preview"
        ] = results[
            "before_scaling"
        ]

    # Model fitting belongs to the dedicated model pages.  Preprocessing ends
    # here so this step stays focused on cleaning, encoding, splitting and scaling.
    return results

    # ========================================================
    # LINEAR REGRESSION
    # ========================================================

    model_results = {}

    MODEL_FEATURE = "CGPA"

    MODEL_TARGET = "Salary Package"

    if (

        MODEL_FEATURE in df.columns

        and

        MODEL_TARGET in df.columns
    ):

        model_df = df[
            [
                MODEL_FEATURE,
                MODEL_TARGET
            ]
        ].copy()

        model_df[
            MODEL_FEATURE
        ] = to_numeric_clean(
            model_df[
                MODEL_FEATURE
            ]
        )

        model_df[
            MODEL_TARGET
        ] = to_numeric_clean(
            model_df[
                MODEL_TARGET
            ]
        )

        model_df = (
            model_df
            .dropna()
        )

        if len(model_df) >= 10:

            # ------------------------------------------------
            # X / Y
            # ------------------------------------------------

            x = model_df[
                MODEL_FEATURE
            ].to_numpy(
                dtype=float
            )

            y = model_df[
                MODEL_TARGET
            ].to_numpy(
                dtype=float
            )

            # ------------------------------------------------
            # TRAIN / VALIDATION
            # ------------------------------------------------

            (

                x_train_model,

                x_val_model,

                y_train_model,

                y_val_model

            ) = train_test_split(

                x,

                y,

                test_size=0.20,

                random_state=42
            )

            # ------------------------------------------------
            # STANDARDIZE X
            # ------------------------------------------------

            (

                x_train_std,

                x_mean,

                x_std

            ) = standardize(
                x_train_model
            )

            if x_std == 0:

                x_std = 1.0

            x_val_std = (

                x_val_model
                - x_mean
            ) / x_std

            # ------------------------------------------------
            # GRADIENT DESCENT
            # ------------------------------------------------

            gd_alpha = 0.05

            gd_epochs = 6

            (

                gd_theta0,

                gd_theta1,

                mse_history

            ) = batch_gradient_descent(

                x_train_std,

                y_train_model,

                alpha=gd_alpha,

                epochs=gd_epochs
            )

            # ------------------------------------------------
            # GD PREDICTIONS
            # ------------------------------------------------

            y_val_gd_pred = (

                gd_theta0

                + gd_theta1
                * x_val_std
            )

            # ------------------------------------------------
            # GD METRICS
            # ------------------------------------------------

            gd_mse = (
                mean_squared_error(
                    y_val_model,
                    y_val_gd_pred
                )
            )

            gd_rmse = np.sqrt(
                gd_mse
            )

            gd_mae = (
                mean_absolute_error(
                    y_val_model,
                    y_val_gd_pred
                )
            )

            gd_r2 = r2_score(
                y_val_model,
                y_val_gd_pred
            )

            # ------------------------------------------------
            # ORIGINAL COEFFICIENTS
            # ------------------------------------------------

            gd_original_slope = (

                gd_theta1
                / x_std
            )

            gd_original_intercept = (

                gd_theta0

                - (

                    gd_theta1
                    * x_mean
                    / x_std
                )
            )

            # ------------------------------------------------
            # SCIKIT LEARN
            # ------------------------------------------------

            sk_model = (
                LinearRegression()
            )

            sk_model.fit(

                x_train_model.reshape(
                    -1,
                    1
                ),

                y_train_model
            )

            y_val_sk_pred = (
                sk_model.predict(

                    x_val_model.reshape(
                        -1,
                        1
                    )
                )
            )

            # ------------------------------------------------
            # SKLEARN METRICS
            # ------------------------------------------------

            sk_mse = (
                mean_squared_error(
                    y_val_model,
                    y_val_sk_pred
                )
            )

            sk_rmse = np.sqrt(
                sk_mse
            )

            sk_mae = (
                mean_absolute_error(
                    y_val_model,
                    y_val_sk_pred
                )
            )

            sk_r2 = r2_score(
                y_val_model,
                y_val_sk_pred
            )

            # ------------------------------------------------
            # MODEL RESULTS
            # ------------------------------------------------

            model_results = {

                "feature":
                    MODEL_FEATURE,

                "target":
                    MODEL_TARGET,

                "train_rows":
                    int(
                        len(
                            x_train_model
                        )
                    ),

                "validation_rows":
                    int(
                        len(
                            x_val_model
                        )
                    ),

                "alpha":
                    gd_alpha,

                "epochs":
                    gd_epochs,

                "standardization": {

                    "mean":
                        round(
                            float(
                                x_mean
                            ),
                            6
                        ),

                    "std":
                        round(
                            float(
                                x_std
                            ),
                            6
                        )
                },

                "gradient_descent": {

                    "theta0":
                        round(
                            float(
                                gd_theta0
                            ),
                            6
                        ),

                    "theta1":
                        round(
                            float(
                                gd_theta1
                            ),
                            6
                        ),

                    "original_intercept":
                        round(
                            float(
                                gd_original_intercept
                            ),
                            6
                        ),

                    "original_slope":
                        round(
                            float(
                                gd_original_slope
                            ),
                            6
                        ),

                    "mse":
                        round(
                            float(
                                gd_mse
                            ),
                            6
                        ),

                    "rmse":
                        round(
                            float(
                                gd_rmse
                            ),
                            6
                        ),

                    "mae":
                        round(
                            float(
                                gd_mae
                            ),
                            6
                        ),

                    "r2":
                        round(
                            float(
                                gd_r2
                            ),
                            6
                        )
                },

                "sklearn": {

                    "intercept":
                        round(
                            float(
                                sk_model.intercept_
                            ),
                            6
                        ),

                    "slope":
                        round(
                            float(
                                sk_model.coef_[0]
                            ),
                            6
                        ),

                    "mse":
                        round(
                            float(
                                sk_mse
                            ),
                            6
                        ),

                    "rmse":
                        round(
                            float(
                                sk_rmse
                            ),
                            6
                        ),

                    "mae":
                        round(
                            float(
                                sk_mae
                            ),
                            6
                        ),

                    "r2":
                        round(
                            float(
                                sk_r2
                            ),
                            6
                        )
                },

                "mse_history": [

                    {

                        "epoch":
                            item["epoch"],

                        "theta0":
                            round(
                                item["theta0"],
                                6
                            ),

                        "theta1":
                            round(
                                item["theta1"],
                                6
                            ),

                        "mse":
                            round(
                                item["mse"],
                                6
                            )
                    }

                    for item
                    in mse_history
                ]
            }

            # ------------------------------------------------
            # CHART DATA
            # ------------------------------------------------

            model_results[
                "_chart_data"
            ] = {

                "x_train":
                    x_train_model.tolist(),

                "y_train":
                    y_train_model.tolist(),

                "x_val":
                    x_val_model.tolist(),

                "y_val":
                    y_val_model.tolist(),

                "gd_predictions":
                    y_val_gd_pred.tolist(),

                "sk_predictions":
                    y_val_sk_pred.tolist(),

                "x_mean":
                    float(x_mean),

                "x_std":
                    float(x_std),

                "gd_theta0":
                    float(gd_theta0),

                "gd_theta1":
                    float(gd_theta1),

                "mse_history":
                    mse_history
            }

            # ------------------------------------------------
            # MODEL PREVIEW
            # ------------------------------------------------

            results[
                "regression_preview"
            ] = [

                {

                    "CGPA":
                        round(
                            float(
                                x_val_model[i]
                            ),
                            4
                        ),

                    "Actual Salary":
                        round(
                            float(
                                y_val_model[i]
                            ),
                            4
                        ),

                    "GD Prediction":
                        round(
                            float(
                                y_val_gd_pred[i]
                            ),
                            4
                        ),

                    "Sklearn Prediction":
                        round(
                            float(
                                y_val_sk_pred[i]
                            ),
                            4
                        )
                }

                for i in range(
                    min(
                        5,
                        len(x_val_model)
                    )
                )
            ]

        else:

            model_results = {

                "error":
                    "Not enough valid CGPA/Salary Package rows."
            }

    else:

        model_results = {

            "error":
                "CGPA or Salary Package column was not found."
        }

    results[
        "linear_regression"
    ] = model_results

    # ========================================================
    # CHARTS
    # ========================================================

    chart_data = None

    if (

        isinstance(
            model_results,
            dict
        )

        and

        "_chart_data"
        in model_results
    ):

        chart_data = (
            model_results[
                "_chart_data"
            ]
        )

    results[
        "charts"
    ] = generate_regression_charts(
        chart_data
    )

    # ========================================================
    # CHART VERSION
    # ========================================================

    results[
        "chart_version"
    ] = str(
        int(
            pd.Timestamp.now()
            .timestamp()
        )
    )

    # ========================================================
    # REMOVE INTERNAL DATA
    # ========================================================

    if isinstance(
        model_results,
        dict
    ):

        model_results.pop(
            "_chart_data",
            None
        )

    # ========================================================
    # FINAL PRINT
    # ========================================================

    print("=" * 80)

    print(
        "PREPROCESSING COMPLETED"
    )

    print(
        "Missing method:",
        missing_method
    )

    print(
        "Missing before:",
        missing_before
    )

    print(
        "Missing after:",
        missing_after
    )

    print(
        "Rows:",
        len(df)
    )

    print(
        "Columns:",
        len(df.columns)
    )

    print(
        "One-hot columns:",
        len(
            results[
                "onehot_columns"
            ]
        )
    )

    print(
        "Ordinal columns:",
        len(
            results[
                "ordinal_columns"
            ]
        )
    )

    print(
        "Target encoded columns:",
        len(
            results[
                "target_columns"
            ]
        )
    )

    print(
        "Embedding columns:",
        len(
            results[
                "embedding_columns"
            ]
        )
    )

    print(
        "Charts generated:",
        len(
            results[
                "charts"
            ]
        )
    )

    for chart in results["charts"]:

        print(
            "  -",
            chart["filename"]
        )

    print("=" * 80)

    return results


# ============================================================
# GENERATE PREPROCESSED CSV
# ============================================================

PREPROCESSED_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "placement_Dataset_preprocessed.csv"
)


def generate_preprocessed_csv(force=False):
    """
    Build a fully preprocessed version of the raw dataset and save it as
    placement_Dataset_preprocessed.csv next to this file.

    Pipeline (applied to the FULL dataset — no train/test split):
      1. Load raw CSV
      2. Drop StudentID  (unique ID, no signal)
      3. Median imputation for numeric columns
      4. Mode  imputation for categorical columns
      5. IQR clipping of CodingTestScore  (clip to [Q1-1.5*IQR, Q3+1.5*IQR])
      6. Ordinal encoding  → CollegeTier (Tier1=2, Tier2=1, Tier3=0)
                           → CGPA_Tier   (High=2, Mid=1, Low=0)
      7. One-hot encoding  → Gender, City, Stream, Specialisation,
                             Hostel, HistoryOfBacklogs
      8. PlacementStatus   → 1 (placed) / 0 (not placed)
      9. StandardScaler on all numeric columns  (zero-mean, unit-variance)
         so KMeans distance is not dominated by high-magnitude features.

    Returns: path to the saved CSV (str)
    """

    if not force and os.path.exists(PREPROCESSED_PATH):
        return PREPROCESSED_PATH          # already exists, skip re-generation

    df = pd.read_csv(FILE_PATH)

    # ── 1. Drop identifier ──────────────────────────────────
    df.drop(columns=["StudentID"], errors="ignore", inplace=True)

    # ── 2. Median imputation — numeric ──────────────────────
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # ── 3. Mode imputation — categorical ────────────────────
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    # ── 4. IQR clip — CodingTestScore ───────────────────────
    if "CodingTestScore" in df.columns:
        q1  = df["CodingTestScore"].quantile(0.25)
        q3  = df["CodingTestScore"].quantile(0.75)
        iqr = q3 - q1
        df["CodingTestScore"] = df["CodingTestScore"].clip(
            lower=q1 - 1.5 * iqr,
            upper=q3 + 1.5 * iqr
        )

    # ── 5. Ordinal encoding ──────────────────────────────────
    tier_map = {"Tier1": 2, "Tier2": 1, "Tier3": 0}
    cgpa_map = {"High": 2, "Mid": 1, "Low": 0}
    if "CollegeTier" in df.columns:
        df["CollegeTier"] = df["CollegeTier"].map(tier_map).fillna(1)
    if "CGPA_Tier" in df.columns:
        df["CGPA_Tier"] = df["CGPA_Tier"].map(cgpa_map).fillna(1)

    # ── 6. PlacementStatus → binary ─────────────────────────
    if "PlacementStatus" in df.columns:
        placed_map = {
            1: 1, 0: 0,
            "1": 1, "0": 0,
            "Placed": 1, "Not Placed": 0, "NotPlaced": 0,
        }
        df["PlacementStatus"] = (
            df["PlacementStatus"]
            .map(placed_map)
            .fillna(df["PlacementStatus"])
        )
        df["PlacementStatus"] = pd.to_numeric(
            df["PlacementStatus"], errors="coerce"
        ).fillna(0).astype(int)

    # ── 7. One-hot encoding ──────────────────────────────────
    ohe_cols = [
        c for c in ["Gender", "City", "Stream", "Specialisation",
                    "Hostel", "HistoryOfBacklogs"]
        if c in df.columns
    ]
    if ohe_cols:
        try:
            enc = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        except TypeError:
            enc = OneHotEncoder(
                handle_unknown="ignore",
                sparse=False
            )
        ohe_array   = enc.fit_transform(df[ohe_cols].astype(str))
        ohe_columns = enc.get_feature_names_out(ohe_cols).tolist()
        ohe_df      = pd.DataFrame(ohe_array, columns=ohe_columns, index=df.index)
        df.drop(columns=ohe_cols, inplace=True)
        df = pd.concat([df, ohe_df], axis=1)

    # ── 8. StandardScaler on ALL numeric columns ────────────
    # (excludes the new 0/1 OHE columns — they are already unit-scale)
    numeric_now = df.select_dtypes(include=np.number).columns.tolist()
    binary_cols = [c for c in numeric_now if df[c].dropna().isin([0, 1]).all()]
    scale_cols  = [c for c in numeric_now if c not in binary_cols]

    if scale_cols:
        scaler = StandardScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])

    # ── 9. Final numeric conversion & round ─────────────────
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].round(6)

    df.to_csv(PREPROCESSED_PATH, index=False)

    print(f"[preprocessing] Saved preprocessed CSV → {PREPROCESSED_PATH}")
    print(f"  Rows: {len(df)}   Columns: {len(df.columns)}")

    return PREPROCESSED_PATH


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    result = run_preprocessing(
        missing_method="median"
    )

    print(
        "\nCharts:"
    )

    for chart in result.get(
        "charts",
        []
    ):

        print(
            "-",
            chart["title"]
        )
