from sklearn.ensemble import GradientBoostingClassifier
from model_training import RANDOM_STATE, run_training

DETAILS = {"title": "Gradient Boosting", "icon": "🚀", "kind": "classification", "description": "Builds trees sequentially, each one fitting the pseudo-residuals (negative gradient of the loss) of the ensemble so far. Very accurate on tabular data; sensitive to learning rate."}

def run():
    return run_training("gradient-boosting", DETAILS, GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=4, min_samples_leaf=10, subsample=0.8, random_state=RANDOM_STATE))
