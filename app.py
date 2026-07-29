from flask import Flask, render_template
from load_data import load_data, get_data_summary

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", active="none")


@app.route("/data-loading")
def data_loading():

    summary = None
    error = None

    try:
        df = load_data()                  # Load the dataset
        summary = get_data_summary(df)    # Generate summary

    except FileNotFoundError as e:
        error = str(e)

    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "index.html",
        active="data-loading",
        summary=summary,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)