import sys
import os 
from train_test_split import *
import datasets
from features import *
from train import *
from test import *
from config import *

def main():
    folder_path = DATASET_NAME
    dir_path = os.path.dirname(os.path.abspath(__file__))
    dir_path = os.path.join(dir_path, "..", "datasets", folder_path)
    if (PREPROCESS):
        print("Started Pre-processing")
        datasets.dataset_selection(dir_path)
        print("Splitting dataset into Train and Test")
        split_and_store_data(dir_path)
    print("Extracting features")
    trainSplit = extract_features(dir_path,'train')
    testSplit = extract_features(dir_path, 'test')
    train_test = TRAIN_TEST.lower()
    model_name = MODEL_NAME
    if train_test == 'train':
        train_model(trainSplit, model_name)
    else:
        # test_model(testSplit)
        raise Exception(f"Not supported {train_test}")
    #test_model(testSplit)


if __name__ == "__main__":
    main()