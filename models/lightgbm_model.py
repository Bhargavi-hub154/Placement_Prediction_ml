import lightgbm as lgb
from model_training import RANDOM_STATE, run_training

DETAILS = {"title": "LightGBM", "icon": "💡", "kind": "classification", "description": "Microsoft's gradient boosting framework. Uses leaf-wise growth and histogram-based splits for 10–20× faster training than standard GBDT while matching or beating accuracy."}

def run():
    return run_training("lightgbm", DETAILS, lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, num_leaves=40, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, verbose=-1))
