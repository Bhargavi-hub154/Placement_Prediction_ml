from sklearn.linear_model import LogisticRegression
from model_training import RANDOM_STATE, run_training

DETAILS = {"title": "Logistic Regression", "icon": "🎯", "kind": "classification", "description": "Models the probability of placement with a sigmoid function. Fast, interpretable baseline for binary classification."}

def run():
    return run_training("logistic", DETAILS, LogisticRegression(max_iter=1200, random_state=RANDOM_STATE))
