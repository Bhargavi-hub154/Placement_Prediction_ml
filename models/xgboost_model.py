import xgboost as xgb
from model_training import RANDOM_STATE, run_training

DETAILS = {"title": "XGBoost", "icon": "🔥", "kind": "classification", "description": "Extreme Gradient Boosting. Adds L1/L2 regularisation, handles missing values natively, and uses second-order gradient information for highly stable, competition-winning performance."}

def run():
    return run_training("xgboost", DETAILS, xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0))
