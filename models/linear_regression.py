from sklearn.linear_model import LinearRegression
from model_training import run_training

DETAILS = {"title": "Linear Regression", "icon": "📈", "kind": "regression", "description": "Predicts continuous salary package using a linear combination of student features. Optimises the sum of squared residuals."}

def run():
    return run_training("linear", DETAILS, LinearRegression())
