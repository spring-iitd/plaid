import os
import pandas as pd
import numpy as np

ch_data_folder = '../data/Car Hacking Dataset/'
ch_dos_data_path = os.path.join(ch_data_folder, 'DoS_dataset.csv')

mcan_data_folder = '../data/M-CAN Intrusion Dataset/'
mcan_dos_data_path = os.path.join(mcan_data_folder, 'g80_mcan_ddos_data.csv')


def shift_columns(df):
    
    for dlc in [2,5,6]:

        df.loc[df['DLC'] == dlc, df.columns[3:]] = df.loc[df['DLC'] == dlc, df.columns[3:]].shift(periods=8-dlc, axis='columns', fill_value='00')

    return df
    

def read_ch_data(data_path):
    
    columns = ['Timestamp','ID', 'DLC', 'data0', 'data1', 'data2', 'data3', 'data4', 
           'data5', 'data6', 'data7', 'label']
    
    data = pd.read_csv(data_path, names = columns)

    data = shift_columns(data)
    
    ##Replacing all NaNs with '00' 
    data = data.replace(np.NaN, '00')
    
    ##Joining all data columns to put all data in one column
    data_cols = ['data0', 'data1', 'data2', 'data3', 'data4', 'data5', 'data6', 'data7']
    
    ##The data column is in hexadecimal
    data['Payload'] = data[data_cols].apply(''.join, axis=1)
    data.drop(columns = data_cols, inplace = True, axis = 1)
        
    data = data.assign(IAT=data['Timestamp'].diff().fillna(0))
    data = data[['Timestamp', 'ID','DLC','Payload', 'IAT', 'label']]
    data['label'].replace({'R' : 0, 'T' : 1}, inplace = True)
    
    return data

def read_mcan_data(data_path):
    
    data = pd.read_csv(data_path)

    ##Replacing all NaNs with '00' 
    data = data.replace(np.NaN, '00')
    
    data['Payload'] = data['Payload'].str.replace(' ', '')
    
    data = data.assign(IAT=data['Timestamp'].diff().fillna(0))
    data = data[['Timestamp', 'ID','DLC','Payload', 'IAT', 'label']]

    return data
    

print("Preprocessing Car Hacking DOS Dataset")
ch_dos_data = read_ch_data(ch_dos_data_path)
ch_dos_data.to_csv(os.path.join(ch_data_folder, 'preprocessed_car_hacking.csv'), index=False)
print("Preprocessed Car Hacking DOS Data Saved")


print()

print("Preprocessing MCAN DOS Dataset")
dos_data = read_mcan_data(mcan_dos_data_path)
dos_data.to_csv(os.path.join(mcan_data_folder, 'preprocessed_mcan.csv'), index=False)
print("Preprocessed MCAN DOS Data Saved")

