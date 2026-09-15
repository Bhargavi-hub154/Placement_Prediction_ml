from sklearn.ensemble import RandomForestClassifier
from model_training import RANDOM_STATE, run_training

DETAILS = {"title": "Random Forest", "icon": "🌳", "kind": "classification", "description": "Bagging: trains N independent trees on bootstrap samples, each considering only a random feature subset at every split. Final prediction is the majority vote — reduces variance."}

def run():
    return run_training("random-forest", DETAILS, RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=5, max_features="sqrt", bootstrap=True, random_state=RANDOM_STATE, n_jobs=-1))
