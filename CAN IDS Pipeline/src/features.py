from utilities import *
import os 
import pandas as pd


def extract_features(dataset_path,train_test):
    modified_dataset_path = os.path.join(dataset_path,train_test)
    df_list = []
    file_paths = [os.path.join(modified_dataset_path, f) for f in os.listdir(modified_dataset_path)]
    for file_path in file_paths:
        df_list.append(read_attack_data(file_path))
    return pd.concat(df_list,axis=0)
      
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