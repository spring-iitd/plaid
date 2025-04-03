import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

from datasets import CANBusDataset, FeatureExtractedDataset
from utilities import *
from preprocessing import *

class FeatureExtractor:
    def __init__(self, datasets):
        """Initialize the FeatureExtractor with the provided parameters.

        :param dataset: The dataset from which to extract features.
        :param selected_features: List of selected feature names to extract.

        """
        self.datasets = datasets
        self.selected_features = self.dataset.get_feature_names()
        self.feature_indices = [self.dataset.get_feature_names().index(f) for f in self.selected_features]


    def get_feature_names(self):
        """
        Get the names of the features, including custom feature names if a custom function is provided.

        :return: List of feature names.
        """
        # Get the names of the selected features
        feature_names = [self.dataset.get_feature_names()[i] for i in self.feature_indices]
        
        return feature_names

    def extract_features(self):
        
        for dataset in self.datasets:
            
            if(dataset.endswith(".txt")):   # Normal run data (".txt")
                dataset.data['can_id'] = self.data['can_id'].apply(hex_to_dec)
                dataset.data['dlc'].astype(int)
                dataset.data['data'] = dataset.data.iloc[:, 3:-1].apply(lambda x: int(''.join(x), 16), axis=1)
                dataset.data.sort_values(by = ['timestamp'], inplace = True)
                dataset.data['IAT'] = dataset.data['timestamp'].diff().fillna(0)
                dataset.data['flag'] = 0
                dataset.data.drop(columns=['data0','data1','data2','data3','data4','data5','data6','data7','flag'])
                dataset.data = self.__read_attack_data(self.data)
            elif(dataset.endswith(".csv")):
                dataset.data = self.__read_attack_data(dataset)   # Car Hacking dataset (".csv")
            elif(dataset.endswith(".log")):
                pass
        dataset.data = self.__merge_data()


    def __read_attack_data(datset):

        data = shift_columns(datset.data)

        ##Replacing all NaNs with '00'
        data = data.replace(np.nan, '00')

        ##Joining all data columns to put all data in one column
        data_cols = ['data0', 'data1', 'data2', 'data3', 'data4', 'data5', 'data6', 'data7']

        #The data column is in hexadecimal
        data['data'] = data[data_cols].apply(''.join, axis=1)
        data.drop(columns = data_cols, inplace = True, axis = 1)

        #Converting columns to decimal
        data['can_id'] = data['can_id'].apply(hex_to_dec)
        data['data'] = data['data'].apply(hex_to_dec)

        data = data.assign(IAT=data['timestamp'].diff().fillna(0))
        #data.to_csv("preprocessed.csv", index=False)  


        return data

    

    
    


if __name__ == "__main__":
    # Assuming you have already created your CANBusDataset
    can_dataset = CANBusDataset('file.csv')

    # Create a FeatureExtractor
    feature_extractor = FeatureExtractor(
        dataset=can_dataset,
        selected_features=['timestamp','can_id', 'dlc', 'data0', 'data1', 'data2', 'data3', 'data4', 'data5', 'data6', 'data7', 'flag']
    )

    # Extract features
    extracted_dataset = feature_extractor.extract_features()

    print("Number of samples:", len(extracted_dataset))
    print("Feature names:", extracted_dataset.get_feature_names())

    # Example of using the extracted dataset with a DataLoader
    extracted_dataloader = DataLoader(extracted_dataset, batch_size=32, shuffle=True)

    for batch_features, batch_labels in extracted_dataloader:
        print("Batch features shape:", batch_features.shape)
        print("Batch labels shape:", batch_labels.shape)
        break  # Just print the first batch