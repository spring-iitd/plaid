from ids.base import IDS
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


class RandomForest(IDS):
    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=100, max_depth=4)

    def train(self, X_train, Y_train, **kwargs):
        self.rf.fit(X_train, Y_train)

    def test(self, X_test, Y_test):
        Y_pred = self.predict(X_test)
        return accuracy_score(Y_test, Y_pred)

    def save(self, path):
        joblib.dump(self.rf, path)

    def predict(self, X_test):
        dt_preds = self.rf.predict(X_test)
        return dt_preds

    def load(self, path):
        self.rf = joblib.load(path)
