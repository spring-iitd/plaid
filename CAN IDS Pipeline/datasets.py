import os
import pandas as pd
import csv
import torch
from torch.utils.data import Dataset
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
            return txtFile_to_CANbusData(file_path)
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



def dataset_selection(dataset_path):
    file_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(('.csv', '.txt','.log'))]
    for file_path in file_paths:
        CANBusDataset(file_path)
