import re
from utilities import *
import pandas as pd
import numpy as np
import os 

curr_dir_path = os.path.dirname(os.path.abspath(__file__))

def MIRGU_to_CANbusData(file_path):

    dir_path = "/".join(file_path.split("/")[:-1])
    mod_dir_path = os.path.join(dir_path,"modified_dataset")
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
    mod_dir_path = os.path.join(dir_path,"modified_dataset")
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
    mod_dir_path = os.path.join(dir_path,"modified_dataset")
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
    mod_dir_path = os.path.join(dir_path,"modified_dataset")
    file = file_path.split("/")[-1].replace(".txt",".csv") 
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
            output_line = f"{timestamp},{can_id},{dlc},{','.join(data)}\n"
            outfile.write(output_line)

    return csv_file
