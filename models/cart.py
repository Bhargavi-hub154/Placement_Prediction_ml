from sklearn.tree import DecisionTreeClassifier
from model_training import RANDOM_STATE, run_training

DETAILS = {"title": "Decision Tree — CART", "icon": "🌲", "kind": "classification", "description": "Uses Gini Impurity as the split criterion. CART always produces binary splits and is the building block for ensemble methods."}

def run():
    return run_training("cart", DETAILS, DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_leaf=10, random_state=RANDOM_STATE))
