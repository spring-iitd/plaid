import os
from sklearn.preprocessing import StandardScaler

from get_ids import get_model

def test_model(TestSplit, modelName, modelPath):
    X_test, Y_test = TestSplit.drop(columns = ['flag', 'timestamp']).values, TestSplit['flag'].values
    
    scaler = StandardScaler()
    scaler.fit(X_test)

    # Transform train and test sets
    X_test = scaler.transform(X_test)

    model = get_model(modelName)
    print(f"Loading model from {os.path.normpath(modelPath)}")
    model.load(modelPath)
    
    print("Test Accuracy: ", model.test(X_test, Y_test))

    print("Testing Completed")