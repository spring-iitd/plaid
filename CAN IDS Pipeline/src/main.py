import os 
from train_test_split import *
from preprocessing import *
from features import *
from train import *
from test import *
from config import *

def main():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(dir_path, "..", "datasets", DATASET_NAME)
    if (PREPROCESS):
        print("Started Pre-processing")
        preprocess(dataset_path)
        print("Splitting dataset into Train and Test")
        split_and_store_data(dataset_path)
    print("Extracting features")
    trainSplit = extract_features(dataset_path,'train')
    testSplit = extract_features(dataset_path, 'test')
    train_test = TRAIN_TEST.lower()
    model_path = os.path.join(dir_path, "..", "models", MODEL_NAME)
    if train_test == 'train':
        train_model(trainSplit, MODEL_NAME, model_path)
    elif train_test == 'test':
        pass
    else:
        raise Exception(f"Not supported {train_test}")
    test_model(testSplit, MODEL_NAME, model_path)


if __name__ == "__main__":
    main()