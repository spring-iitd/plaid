from sklearn.preprocessing import StandardScaler

from get_ids import get_model

def train_model(TrainSplit,modelName):
    if(modelName == "MLP"):
        X_train, Y_train = TrainSplit.drop(columns = ['flag', 'timestamp']).values, TrainSplit['flag'].replace({'R': 0, 'T': 1}).values
    else:
        X_train, Y_train = TrainSplit.drop(columns = ['flag', 'timestamp']).values, TrainSplit['flag'].values
    
    scaler = StandardScaler()
    scaler.fit(X_train)

    # Transform train and test sets
    X_train = scaler.transform(X_train)

    model = get_model(modelName)
    
    print("Starting Training")
    model.train(X_train, Y_train)
    print("Training Completed")