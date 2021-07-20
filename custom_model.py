from sklearn.ensemble import AdaBoostClassifier
from sklearn.base import BaseEstimator


class AdaBoostModel(BaseEstimator):
    def __init__(self):
        self.model = AdaBoostClassifier(n_estimators=100,
                                       learning_rate=0.1,
                                       random_state=0)

    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return None
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
