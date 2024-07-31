import numpy as np
import pandas as pd
import os
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime


data_folder = '../data/Car Hacking Dataset/'

def hex_to_bin(hex_num):
    
    binary_value = bin(int(str(hex_num), 16))[2:]
    
    return binary_value

def int_to_bin(int_num):
    
    binary_value = bin(int_num)[2:]
    
    return binary_value

hex_to_dec = lambda x: int(x, 16)
dec_to_hex = lambda x : hex(int(x))[2:]

def read_data(data_path):
    
    columns = ['Timestamp','ID', 'DLC', 'Payload', 'label']
    
    data = pd.read_csv(data_path)
    
    ##Replacing all NaNs with '00' 
    data = data.replace(np.NaN, '00')

    data['ID'] = data['ID'].apply(hex_to_dec)
    
    data['Payload'] = data['Payload'].str.replace(' ', '')
    data['Payload'] = data['Payload'].apply(hex_to_dec)
    
    data = data.assign(IAT=data['Timestamp'].diff().fillna(0))
    
    return data

normal_data = read_data(os.path.join(data_folder, 'benign_data.csv'))

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

def injection_possible(inter_arrival_time):

    #This is the time that it would take a fully stuffed frame to be transmitted 
    #to the bus at 512kbps 
    time_to_transit = 0.000255

    num_injections = (inter_arrival_time - time_to_transit) // time_to_transit

    if num_injections >= 1:
        return (True, num_injections)
    else:
        return (False, 0)
    
def calculate_atr(data, ts, periodicity_dict):

    atr_dict = dict()

    """data : The dataflow until the time at which decision to inject packet is taken"""

    ids = list(periodicity_dict.keys())

    last_occurrences = data.drop_duplicates(subset='ID', keep='last')

    for id in ids:

        last_occurrence = last_occurrences[last_occurrences['ID'] == id]['Timestamp']
        
        atr = (ts - last_occurrence)/periodicity_dict[id]

        if len(atr.values) == 0:
            atr = 0.0
        else:
            atr = atr.item()
    
        atr_dict[id] = atr
            
    return atr_dict

## Threshold same or different for different IDs?


##Assuming we know when the next packet comes using periodicity dict/ lookahead,
## In actual it will shift packets or we change approach?

def attack(data, thresh):

    data = data.drop(columns=['IAT'],axis = 1)
    standby_packets = 100_000

    out = []

    for ind in range(len(data)):
        
        print(f"{ind} out of {len(data)}: {ind/len(data)*100}%")

        if ind <= standby_packets:
            out.append(data.iloc[ind].values)
        
        else:
            curr_ts = data['Timestamp'][ind]
            prev_ts = data['Timestamp'][ind - 1]

            curr_iat = curr_ts - prev_ts

            if injection_possible(curr_iat)[0]:
            
                periodicity_dict = calculate_periodicity(data[:ind])

                atr_dict =  calculate_atr(data[:ind], curr_ts, periodicity_dict)
            
                attack_id = max(atr_dict, key=atr_dict.get)

                min_thresh = thresh * periodicity_dict[attack_id]

                last_encountered_ts = data[:ind][data[:ind]['ID'] == attack_id].iloc[-1]['Timestamp']

                lb1 = last_encountered_ts + min_thresh
                lb2 = prev_ts + 0.000255

                lower_bound = max(lb1, lb2)

                if injection_possible(curr_ts - lower_bound):

                    rand_data = data[:ind][data[:ind]['ID'] == attack_id].sample(1)

                    attack_frame = [lower_bound + 0.000255, attack_id, rand_data['DLC'], rand_data['Payload'], 1]

                    out.append(attack_frame)
                
            out.append(data.iloc[ind].values)

    print("\n\n\n\n")

    return out


thresh_vals = [x * 0.01 for x in range(5,95,5)]

output_dir = "../data/New Attack/"


for val in thresh_vals:
    data_out = attack(normal_data)

    print(f"Threshold value: {val}")
    
    out_df = pd.DataFrame(data_out, columns = ['Timestamp','ID', 'DLC', 'Payload', 'label'])

    print(f"Packets injected: {out_df['label'].value_counts()[1]}")
    
    out_path = os.path.join(output_dir, f"thresh_val_{val}")
    
    out_df.to_csv(out_path)





# standby_packets = 20_000

# threshold = 0.5


# out = []

# for ind in range(len(normal_data)):
    
#     print(f"{ind} out of {len(normal_data)}: {ind/len(normal_data)*100}%")

#     if ind <= standby_packets:
#         out.append(normal_data.iloc[ind].values)
    
#     else:
#         curr_ts = normal_data['Timestamp'][ind]
#         prev_ts = normal_data['Timestamp'][ind - 1]

#         curr_iat = curr_ts - prev_ts

#         if injection_possible(curr_iat)[0]:
           
#             periodicity_dict = calculate_periodicity(normal_data[:ind])

#             atr_dict =  calculate_atr(normal_data[:ind], curr_ts, periodicity_dict)
          
#             attack_id = max(atr_dict, key=atr_dict.get)

#             min_thresh = threshold * periodicity_dict[attack_id]

#             last_encountered_ts = normal_data[:ind][normal_data[:ind]['ID'] == attack_id].iloc[-1]['Timestamp']

#             lb1 = last_encountered_ts + min_thresh
#             lb2 = prev_ts + 0.000255

#             lower_bound = max(lb1, lb2)

#             if injection_possible(curr_ts - lower_bound):

#                 rand_data = normal_data[:ind][normal_data[:ind]['ID'] == attack_id].sample(1)

#                 attack_frame = [lower_bound + 0.000255, attack_id, rand_data['DLC'], rand_data['Payload'], 1]

#                 out.append(attack_frame)
            
#         out.append(normal_data.iloc[ind].values)


    