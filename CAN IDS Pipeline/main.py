import sys
import os 
from train_test_split import *
import datasets
from features import *


def main():
    folder_path = input("Dataset folder path : ")
    dir_path = os.path.dirname(os.path.abspath(__file__))
    dir_path = os.path.join(dir_path,"datasets",folder_path)
    datasets.dataset_selection(dir_path)
    split_and_store_data(dir_path)
    train_test = input("Train/Test : ").lower()
    features = extract_features(dir_path,train_test)


if __name__ == "__main__":
    main()