import numpy as np
import pandas as pd
import os
from collections import defaultdict
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('directory', 
                    type=str, 
                    help = 'directory where files are stored')

parser.add_argument('file_name',
                    type = str,
                    help = 'benign data file')


args = parser.parse_args()

data_folder = args.directory

def hex_to_bin(hex_num):
    
    binary_value = bin(int(str(hex_num), 16))[2:]
    
    return binary_value

def int_to_bin(int_num):
    
    binary_value = bin(int_num)[2:]
    
    return binary_value

hex_to_dec = lambda x: int(x, 16)

def read_data(data_path):
    
    columns = ['Timestamp','ID', 'DLC', 'Payload', 'label']
    
    data = pd.read_csv(data_path)
    
    ##Replacing all NaNs with '00' 
    data = data.replace(np.NaN, '00')

    data['ID'] = data['ID'].apply(hex_to_dec)
    
    data['Payload'] = data['Payload'].str.replace(' ', '')
    data['Payload'] = data['Payload'].apply(hex_to_dec)
    
    # data = data.assign(IAT=data['Timestamp'].diff().fillna(0))
    
    return data

normal_data = read_data(os.path.join(data_folder, args.file_name))

## Function to calculate average time and frequency of each message ID
## In this function frequency is defined as the number of packets a particular message ID has
## Average time is defined as the average time between two packets of the same message ID
def calculate_average_time_frequency(dataframe):
    
    frame_times = defaultdict(list)
    total_frames = len(dataframe)
    id_frequency = dataframe["ID"].value_counts()
    id_frequency = defaultdict(int, id_frequency)

    for index, row in dataframe.iterrows():
        id_value = row["ID"]
        timestamp_value = row["Timestamp"]
        frame_times[id_value].append(timestamp_value)


    average_times = {}
    for arbitration_id, timestamps in frame_times.items():
        if len(timestamps) > 1:
            differences = [(timestamps[i+1] - timestamps[i]) for i in range(len(timestamps)-1)]
            average_times[arbitration_id] = sum(differences) / len(differences)
        else:
            average_times[arbitration_id] = 0

    return average_times, id_frequency, total_frames

def calculate_periodicity(df):
    # Initialize an empty dictionary to store the periodicity for each can_id
    periodicity_dict = {}

    # Group the DataFrame by 'can_id'
    grouped = df.groupby('ID')

    for can_id, group_df in grouped:
        # Sort the DataFrame by 'timestamp'
        sorted_df = group_df.sort_values(by='Timestamp')

        # Calculate the periodicity for the current can_id
        first_timestamp = sorted_df.iloc[0]['Timestamp']
        last_timestamp = sorted_df.iloc[-1]['Timestamp']
        num_occurrences = len(sorted_df)

        if num_occurrences > 1:
            periodicity = (last_timestamp - first_timestamp) / (num_occurrences - 1)
        else:
            periodicity = 0  # Avoid division by zero if there's only one occurrence

        # Store the periodicity in the dictionary
        periodicity_dict[can_id] = periodicity

    return periodicity_dict

def calculate_inverse_periodicity(periodicity_dict):
    
    inverse_periodicity_dict = {id: 1.0 / periodicity for id, periodicity in periodicity_dict.items()}
    
    return inverse_periodicity_dict

periodicity_dict = calculate_periodicity(normal_data)

##Latest function
def add_dos(data, top_id_to_consider = 5):
    
    ## This is the time in seconds that the bus would take to transmit a packet which has 110 bits (maximum packet size)
    max_time_to_transmit = 0.00021
    
    periodicity_dict = calculate_periodicity(data)
    sorted_periodicity_dict = dict(sorted(periodicity_dict.items(), key=lambda item: item[1], reverse=True))

    ## Retrieving message ID with highest periodicity, this will be the message ID that we will use to inject packets
    highest_periodicity_id = list(sorted_periodicity_dict.keys())[0]
    highest_periodicity =  list(sorted_periodicity_dict.values())[0]

    ## Retrieving message ID with lowest periodicity, our target message ID shouldnt have a lower periodicity that this after injection is complete
    lowest_periodicity_id = list(sorted_periodicity_dict.keys())[-1]
    lowest_periodicity =  list(sorted_periodicity_dict.values())[-1]
    
    ## Retrieving the indexes at which there is a message with highest periodicity message ID
    index_list_of_highest_arb_id = list(data[data['ID'] == highest_periodicity_id].index)
    
    ##TODO: currently only 1 packet is being inserted, choose frequency difference / lowest frequency
    ##TODO: Randomize attack packet arb id based on frequency : If injecting multiple packets at once, they can be of different IDs and follow the same attack principle
    

    out_data = []
    packets_inserted = 0
    pointer = 0 
    offset = 0.00021 #make this dynamic depending on attack packet size
    
    for index in tqdm(range(len(index_list_of_highest_arb_id) - 1)):

        ts1 = data['Timestamp'].iloc[index_list_of_highest_arb_id[index]]
        ts2 = data['Timestamp'].iloc[index_list_of_highest_arb_id[index + 1]]
        
        last_encountered_ts = ts1

        while last_encountered_ts < ts2:

            # print(ts1, last_encountered_ts, ts2)  
            
            matching_ts = last_encountered_ts + lowest_periodicity

            
            matching_packet_idx = data[data['Timestamp'] < matching_ts].index.max()
            interval = data['Timestamp'].iloc[matching_packet_idx+1] - data['Timestamp'].iloc[matching_packet_idx]

            num_packets_to_insert = int((interval - 0.00021) // 0.00021)

            if num_packets_to_insert >= 1:

                # Timestamp	ID	DLC	Payload	label

                out_data.extend((data.iloc[pointer:matching_packet_idx]).values)
                attack_packet = [data['Timestamp'].iloc[matching_packet_idx] + lowest_periodicity, highest_periodicity_id, 8, np.random.choice(data[data['ID'] == highest_periodicity_id]['Payload']),1]
                out_data.append(attack_packet)

                for packets in range(1, num_packets_to_insert):
                    rand_id = np.random.choice(list(sorted_periodicity_dict.keys())[:top_id_to_consider])
                    attack_packet = [data['Timestamp'].iloc[matching_packet_idx] + (packets + 1) * lowest_periodicity, rand_id, 8,  np.random.choice(data[data['ID'] == rand_id]['Payload']),1]
                    out_data.append(attack_packet)
                

                packets_inserted += num_packets_to_insert
    
                pointer = matching_packet_idx 
                last_encountered_ts = data['Timestamp'].iloc[matching_packet_idx] + lowest_periodicity
            
            else:
                last_encountered_ts += lowest_periodicity 
        
    out_data.extend((data.iloc[pointer:]).values)
    print(f"Packets inserted: {packets_inserted}")
    return out_data
        
        
out = add_dos(normal_data)


out_df = pd.DataFrame(out, columns = ['Timestamp','ID', 'DLC', 'Payload', 'label'])


output_file = os.path.join(data_folder, 'smart_output.csv')

out_df.to_csv(output_file, index = False)