from sklearn.ensemble import IsolationForest
import numpy as np
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'isolation_forest.joblib')

class AnomalyDetector:
    def __init__(self):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
        else:
            self.model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)

    def fit(self, X):
        self.model.fit(X)
        joblib.dump(self.model, MODEL_PATH)

    def predict(self, X):
        return self.model.predict(X), self.model.decision_function(X)

    def is_trained(self):
        return hasattr(self.model, 'estimators_')

detector = AnomalyDetector() 