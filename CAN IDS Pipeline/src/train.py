import os

from get_ids import get_model

def train_model(TrainSplit, modelName, modelPath):
    X_train, Y_train = TrainSplit.drop(columns = ['flag', 'timestamp']).values, TrainSplit['flag'].values
    
    model = get_model(modelName)
    
    print("Starting Training")
    model.train(X_train, Y_train)
    print("Training Completed")

    model.save(modelPath)
    print(f"Model saved at {os.path.normpath(modelPath)}")