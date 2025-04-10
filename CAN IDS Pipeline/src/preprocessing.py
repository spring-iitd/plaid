import re
from utilities import *
import pandas as pd
import numpy as np
import os 
import sys
import importlib
from torch.utils.data import Dataset

def extract_features(dataset_path,train_test):
    modified_dataset_path = os.path.join(dataset_path,"modified_dataset",train_test)
    df_list = []
    file_paths = [os.path.join(modified_dataset_path, f) for f in os.listdir(modified_dataset_path)]
    for file_path in file_paths:
        df_list.append(read_attack_data(file_path))
      
      
def read_attack_data(data_path):

    columns = ['timestamp','can_id', 'dlc', 'data0', 'data1', 'data2', 'data3', 'data4',
        'data5', 'data6', 'data7', 'flag']

    data = pd.read_csv(data_path, names = columns,skiprows=1)
    data = shift_columns(data)

    ##Replacing all NaNs with '00'
    data = data.replace(np.NaN, '00')

    ##Joining all data columns to put all data in one column
    data_cols = ['data0', 'data1', 'data2', 'data3', 'data4', 'data5', 'data6', 'data7']

    ##The data column is in hexadecimal
    data['data'] = data[data_cols].apply(''.join, axis=1)
    data.drop(columns = data_cols, inplace = True, axis = 1)

    ##Converting columns to decimal
    data['can_id'] = data['can_id'].apply(hex_to_dec)
    data['data'] = data['data'].apply(hex_to_dec)

    data = data.assign(IAT=data['timestamp'].diff().fillna(0))

    return data

def MIRGU_to_CANbusData(file_path):

    dir_path = "/".join(file_path.split("/")[:-1])
    mod_dir_path = os.path.join(dir_path, "..","modified_dataset")
    file = file_path.split("/")[-1][:-4]+".csv"
    os.makedirs(mod_dir_path, exist_ok=True)
    csv_file = os.path.join(mod_dir_path,file)
    # Open the input file and output file
    with open(file_path, 'r') as infile, open(csv_file, 'w') as outfile:
        # Read each line in the input file
        for line in infile:
            # Match the pattern for a CAN log line
            match = re.match(r'\((\d+\.\d+)\)\s+can0\s+([0-9A-Fa-f]+)#([0-9A-Fa-f]+)\s+(\d)', line)

            if match:
                # Extract the components from the matched line
                timestamp = match.group(1)  # Timestamp
                can_id = match.group(2)     # CAN ID
                data = match.group(3)       # Data field in hexadecimal
                status = match.group(4)     # Status (0 or 1)

                # Add leading zero to CAN ID to make it 4 digits
                can_id = can_id.zfill(4)

                # Calculate DLC based on the length of the data field (each byte is represented by 2 hex characters)
                data_length = len(data) // 2  # Divide by 2 because each byte is represented by 2 hex characters
                dlc = data_length  # DLC is the number of bytes in the data

                # Only add trailing zeros if the DLC is not 8 bytes
                if dlc < 8:
                    # Add trailing zeros to the data to ensure it is 8 bytes (16 characters) long
                    data = data.ljust(16, '0')  # Ensure the data is padded with trailing zeros to 16 hex characters

                # Split data into bytes (each byte is represented by 2 hex characters)
                data_bytes = [data[i:i+2] for i in range(0, len(data), 2)]

                # Label based on status: R if 0, T if 1
                label = 'R' if status == '0' else 'T'

                # Prepare the output format with DLC
                output_line = f"{timestamp},{can_id},{dlc},{','.join(data_bytes)},{label}\n"
                # Write the formatted line to the output file
                outfile.write(output_line)
        return csv_file

def normal_to_CANbusData(file_path):
    dir_path = "/".join(file_path.split("/")[:-1])
    mod_dir_path = os.path.join(dir_path, "..","modified_dataset")
    file = file_path.split("/")[-1].replace(".txt",".csv") 
    os.makedirs(mod_dir_path, exist_ok=True)
    csv_file = os.path.join(mod_dir_path,file)

    # Read the data from the file
    with open(file_path, 'r') as infile, open(csv_file, 'w') as outfile:
        for line in infile:
            # Extract information from each line
            line = line.strip()

            ts = line.split('Timestamp: ')[1].split(' ')[0]
            can_id = line.split('ID: ')[1].split(' ')[0]
            dlc = line.split('DLC: ')[1].split(' ')[0]
            data = line.strip().split()[-int(dlc):]
            # Ensures each row has exactly 8 elements
            data = data + ['00'] * (8 - int(dlc))
            label = 'R'
            output_line = f"{ts},{can_id},{dlc},{','.join(data)},{label}\n"
            # Write the formatted line to the output file
            outfile.write(output_line)
    return csv_file


def CH_to_CANbusData(file_path):
    dir_path = "/".join(file_path.split("/")[:-1])
    mod_dir_path = os.path.join(dir_path, "..","modified_dataset")
    file = file_path.split("/")[-1][:-4]+".csv"
    os.makedirs(mod_dir_path, exist_ok=True)
    csv_file = os.path.join(mod_dir_path,file)

    columns = ['timestamp','can_id', 'dlc', 'data0', 'data1', 'data2', 'data3', 'data4',
           'data5', 'data6', 'data7', 'flag']

    data = pd.read_csv(file_path, names = columns,low_memory=False,skiprows=2)
    #data = shift_columns(data)

    ##Replacing all NaNs with '00'
    data = data.replace(np.nan, '00')
    data.to_csv(csv_file, index=False) 

    return csv_file

def OTIDS_to_CANbusData(file_path):
    dir_path = "/".join(file_path.split("/")[:-1])
    mod_dir_path = os.path.join(dir_path, "..","modified_dataset")
    file = file_path.split("/")[-1].replace(".txt",".csv")

    attack_free = 'attack_free' in file.lower()

    os.makedirs(mod_dir_path, exist_ok=True)
    csv_file = os.path.join(mod_dir_path,file)
    
    with open(file_path, 'r') as input_file, open(csv_file, 'w') as outfile:
        for line in input_file:
            line = line.strip()
            timestamp = float(line.split("Timestamp: ")[1].strip().split(' ')[0])
            can_id = line.split('ID: ')[1].split()[0]
            dlc = int(line.split('DLC: ')[1].split()[0])
            data = [] if dlc == 0 else line.strip().split()[-int(dlc):]
            # Ensures each row has exactly 8 elements
            data = data + ['00'] * (8 - int(dlc))
            label = 'R' if attack_free else ('T' if hex_to_dec(can_id) == 0 else 'R')
            output_line = f"{timestamp},{can_id},{dlc},{','.join(data)},{label}\n"
            outfile.write(output_line)

    return csv_file

def txtFile_to_CANbusData(file_path):
    filename = os.path.basename(file_path)
    if 'normal' in filename.lower():
        return normal_to_CANbusData(file_path)
    else:
        return OTIDS_to_CANbusData(file_path)
    

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



def preprocess_dataset(dataset_path):
    orig_dataset_path = os.path.join(dataset_path, 'original_dataset')
    if not os.path.exists(orig_dataset_path):
        os.makedirs(orig_dataset_path)
        for file in os.listdir(dataset_path):
            if os.path.isdir(file) or not file.endswith(('.csv', '.txt','.log')):
                continue
            old_file_path = os.path.join(dataset_path, file)
            new_file_path = os.path.join(orig_dataset_path, file)
            os.system(f"mv \"{old_file_path}\" \"{new_file_path}\"")
    file_paths = [os.path.join(orig_dataset_path, f) for f in os.listdir(orig_dataset_path) if f.endswith(('.csv', '.txt','.log'))]
    for file_path in file_paths:
        CANBusDataset(file_path)

def preprocess(dataset_path):
    preprocess_file_path = os.path.join(dataset_path, 'preprocess_dataset.py')
    if os.path.exists(preprocess_file_path):
        sys.path.append(dataset_path)
        ppd = importlib.import_module('preprocess_dataset')
        if hasattr(ppd, 'preprocess_dataset'):
            print("Using Preprocessing Script in Dataset")
            return ppd.preprocess_dataset(dataset_path)
    print("Using Default Preprocessing Script")
    preprocess_dataset(dataset_path)