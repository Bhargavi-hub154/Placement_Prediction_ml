from sklearn.tree import DecisionTreeClassifier
from model_training import RANDOM_STATE, run_training

DETAILS = {"title": "Decision Tree — ID3", "icon": "🌿", "kind": "classification", "description": "Builds the tree by maximising Information Gain (entropy reduction) at each split. Produces a fully human-readable tree."}

def run():
    return run_training("id3", DETAILS, DecisionTreeClassifier(criterion="entropy", max_depth=5, min_samples_leaf=10, random_state=RANDOM_STATE))
