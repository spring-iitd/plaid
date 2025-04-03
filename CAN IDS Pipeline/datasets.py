import os
import csv
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import json
from utilities import *
from preprocessing import * 

class CANBusDataset(Dataset):
    def __init__(self, file_path):
        csv_file = self.__preprocess_data(file_path) 
        self.data = self.load_csv(csv_file)
        self.unique_labels = sorted(set(self.data['flag'].astype(str)))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}
        
    def __preprocess_data(self,file_path):
        if(file_path.endswith(".txt")):
            return OTIDS_to_CANbusData(file_path)
        elif(file_path.endswith(".csv")):
            return CH_to_CANbusData(file_path)
        elif(file_path.endswith(".log")):
            return MIRGU_to_CANbusData(file_path)
        else:
            print("INVALID PATH ")

    def load_csv(self, csv_file):
        columns = ['timestamp','can_id', 'dlc', 'data0', 'data1', 'data2', 'data3', 'data4',
            'data5', 'data6', 'data7', 'flag']
        df = pd.read_csv(csv_file, names=columns, header=None,low_memory=False,skiprows = 2)
        return df

    def __len__(self):
        return len(self.data)

    def get_feature_names(self):
        return ['timestamp', 'can_id', 'dlc', 'data0', 'data1','data2', 'data3', 'data4', 'data5', 'data6', 'data7', 'flag']

    def get_num_features(self):
        return len(self.get_feature_names())

    def get_num_classes(self):
        return len(self.unique_labels)
    
    def get_label_mapping(self):
        return self.label_to_idx

    
class FeatureExtractedDataset(Dataset):
    def __init__(self, features, labels, feature_names, label_mapping):
        self.features = features
        self.labels = labels
        self.feature_names = feature_names
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def get_feature_names(self):
        return self.feature_names

    def get_label_mapping(self):
        return self.label_mapping

    def save_dataset(self, directory, features_filename='features.npy', labels_filename='labels.npy', 
                     metadata_filename='metadata.csv', label_mapping_filename='label_mapping.json'):
        os.makedirs(directory, exist_ok=True)
        
        # Save features and labels as numpy arrays
        features_path = os.path.join(directory, features_filename)
        labels_path = os.path.join(directory, labels_filename)
        np.save(features_path, self.features.cpu().numpy())
        np.save(labels_path, self.labels.cpu().numpy())
        
        # Save metadata (feature names)
        metadata_path = os.path.join(directory, metadata_filename)
        with open(metadata_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['feature_name'])
            for feature_name in self.feature_names:
                writer.writerow([feature_name])

        # Save label mapping
        label_mapping_path = os.path.join(directory, label_mapping_filename)
        with open(label_mapping_path, 'w') as jsonfile:
            json.dump(self.label_mapping, jsonfile)

        print(f"Dataset saved to {directory}")

    @classmethod
    def load_dataset(cls, directory, features_filename='features.npy', labels_filename='labels.npy', 
                     metadata_filename='metadata.csv', label_mapping_filename='label_mapping.json'):
        features_path = os.path.join(directory, features_filename)
        labels_path = os.path.join(directory, labels_filename)
        metadata_path = os.path.join(directory, metadata_filename)
        label_mapping_path = os.path.join(directory, label_mapping_filename)
        
        # Load features and labels
        features = torch.from_numpy(np.load(features_path))
        labels = torch.from_numpy(np.load(labels_path))
        
        # Load feature names
        df = pd.read_csv(metadata_path)
        feature_names = df['feature_name'].tolist()
        
        # Load label mapping
        with open(label_mapping_path, 'r') as jsonfile:
            label_mapping = json.load(jsonfile)
        
        return cls(features, labels, feature_names, label_mapping)

    def get_labels(self):
        return self.labels

