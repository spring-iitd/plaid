from ids.base import IDS
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import joblib

class DecisionTree(IDS):
    def __init__(self):
        self.dt = DecisionTreeClassifier(max_depth = 4)

    def train(self, X_train, Y_train, **kwargs):
        self.dt.fit(X_train, Y_train)

    def test(self, X_test, Y_test):
        Y_pred = self.predict(X_test)
        return accuracy_score(Y_test, Y_pred)

    def save(self, path):
        joblib.dump(self.dt, path)

    def predict(self, X_test):
        dt_preds = self.dt.predict(X_test)
        return dt_preds

    def load(self, path):
        self.dt = joblib.load(path)