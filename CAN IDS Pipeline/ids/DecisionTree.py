from ids.base import IDS
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import joblib
from sklearn.preprocessing import StandardScaler

class DecisionTree(IDS):
    def __init__(self):
        self.dt = DecisionTreeClassifier(max_depth = 4)

    def train(self, X_train, Y_train, **kwargs):
        super().train(X_train, Y_train)
        self.dt.fit(self.X, self.Y)

    def test(self, X_test, Y_test):
        super().train(X_test, Y_test)
        Y_pred = self.predict(self.X)
        return accuracy_score(self.Y, Y_pred)

    def save(self, path):
        joblib.dump(self.dt, path)

    def predict(self, X_test):
        super().predict(X_test)
        dt_preds = self.dt.predict(self.X)
        return dt_preds

    def load(self, path):
        self.dt = joblib.load(path)

    def preprocess(self, X, Y):
        scaler = StandardScaler()
        scaler.fit(X)

        # Transform train and test sets
        X = scaler.transform(X)

        return X, Y