import numpy as np
import pandas as pd
import os
from collections import defaultdict
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('directory', 
                    type=str, 
                    help = 'directory where files are stored')

parser.add_argument('file_name',
                    type = str,
                    help = 'benign data file')

parser.add_argument('factor',
                    type = float,
                    help = 'factor to be multiplied for target periodicity')


args = parser.parse_args()

data_folder = args.directory

factor = args.factor

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


def add_dos(data, top_id_to_consider = 5, factor = factor):
    
    ## This is the time in seconds that the bus would take to transmit a packet which has 110 bits (maximum packet size)
    max_time_to_transmit = 0.00021
    
    periodicity_dict = calculate_periodicity(data)
    sorted_periodicity_dict = dict(sorted(periodicity_dict.items(), key=lambda item: item[1], reverse=True))

    ## Retrieving message ID with highest periodicity, this will be the message ID that we will use to inject packets
    highest_periodicity_id = list(sorted_periodicity_dict.keys())[0]
    highest_periodicity =  list(sorted_periodicity_dict.values())[0]

    ## Retrieving message ID with lowest periodicity, our target message ID shouldnt have a lower periodicity than this after injection is complete
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

        ts1 = normal_data['Timestamp'].iloc[index_list_of_highest_arb_id[index]]
        ts2 = normal_data['Timestamp'].iloc[index_list_of_highest_arb_id[index + 1]]

        # print("index_list_of_highest_arb_id[index]):",index_list_of_highest_arb_id[index])
        # print("index_list_of_highest_arb_id[index+1]:",index_list_of_highest_arb_id[index+1])

        last_encountered_ts = ts1

        while last_encountered_ts < ts2:

            # print(ts1, last_encountered_ts, ts2)  
            ## DLC is always 8, data might not be

            ##Adjust this parameter for attack intensity
#             target_periodicity = 0.035 * lowest_periodicity

            target_periodicity = factor * lowest_periodicity

            if target_periodicity <= max_time_to_transmit:
                target_periodicity = lowest_periodicity
            
            #Because we are trying to bring the periodcity of the attack IDs to the lowest
            matching_ts = last_encountered_ts + target_periodicity

            #Finding the location of where attack packet should be if we are trying to match the lowest periodicity
            matching_packet_idx = normal_data[normal_data['Timestamp'] < matching_ts].index.max()
            # print("matching_packet_idx:",matching_packet_idx)

            #No injection packet can have an higher timestamp than this value
            injection_ts_upper_bound = normal_data['Timestamp'].iloc[matching_packet_idx + 1] - max_time_to_transmit
            interval = normal_data['Timestamp'].iloc[matching_packet_idx + 1] - normal_data['Timestamp'].iloc[matching_packet_idx]

            # print(matching_packet_idx)
            num_packets_to_insert = int((interval - max_time_to_transmit) // max_time_to_transmit)

            # print("interval:", interval)

            # print("num_packets_to_insert:",num_packets_to_insert)

            # if counter != 0:
            #     break

            if num_packets_to_insert >= 1:

                # Timestamp	ID	DLC	Payload	label
                out_data.extend((normal_data.iloc[pointer:matching_packet_idx+1]).values)

                attack_ts = normal_data['Timestamp'].iloc[matching_packet_idx] + target_periodicity
                
                if attack_ts >= injection_ts_upper_bound:
                    pointer = matching_packet_idx + 1
                    last_encountered_ts = attack_ts
                    break
                else:
                    attack_packet = [attack_ts, highest_periodicity_id, 8, np.random.choice(normal_data[normal_data['ID'] == highest_periodicity_id]['Payload']),1]
                    out_data.append(attack_packet)
                    packets_inserted += 1

                for packets in range(1, num_packets_to_insert):
                    rand_id = np.random.choice(list(sorted_periodicity_dict.keys())[:5])
                    attack_ts += max_time_to_transmit + target_periodicity
                    
                    if attack_ts >= injection_ts_upper_bound:
                        break
                    else:
                        attack_packet = [attack_ts, rand_id, 8,  np.random.choice(normal_data[normal_data['ID'] == rand_id]['Payload']),1]
                        out_data.append(attack_packet)
                        packets_inserted += 1
                

                

                pointer = matching_packet_idx + 1
                last_encountered_ts = attack_ts
            
            else:
                last_encountered_ts += target_periodicity 

            
    out_data.extend((normal_data.iloc[pointer:]).values)
    print(f"Packets inserted: {packets_inserted}")
    return out_data       
        
out = add_dos(normal_data)


out_df = pd.DataFrame(out, columns = ['Timestamp','ID', 'DLC', 'Payload', 'label'])


def plot_periodicity_df(periodicity_df, save_path):
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.35
    bar_positions = range(len(periodicity_df))

    benign_bars = ax.bar(bar_positions, periodicity_df['Benign Periodicity'], bar_width, label='Periodicity in Benign Data')
    smart_bars = ax.bar([p + bar_width for p in bar_positions], periodicity_df['Smart Periodicity'], bar_width,        label='Periodicity after Adversarial Attack')

    ax.set_xticks([p + bar_width / 2 for p in bar_positions])
    ax.set_xticklabels(periodicity_df['ID'])

    ax.set_xlabel('ID')
    ax.set_ylabel('Periodicity')
    ax.set_title('Benign vs Attack Periodicity')
    ax.legend()

    plt.savefig(save_path)



def plot_outputs(normal_data, out_df, save_loc):

    normal_data = normal_data.assign(IAT=normal_data['Timestamp'].diff().fillna(0))
    out_df = out_df.assign(IAT=out_df['Timestamp'].diff().fillna(0))

    normal_periodicity = calculate_periodicity(normal_data)
#     normal_data['ID'] = normal_data['ID'].apply(hex_to_dec)
    
    smart_periodicity = calculate_periodicity(out_df)
    
    arb_ids = list(normal_periodicity.keys())

    benign_periodicity_values = [normal_periodicity[can_id] for can_id in arb_ids]
    smart_periodicity_values = [smart_periodicity[can_id] for can_id in arb_ids]

    periodicity_df = pd.DataFrame(columns = 
                              ['ID', 'Benign Periodicity', 'Smart Periodicity'],
                              data = {'ID': arb_ids,
                                      'Benign Periodicity': benign_periodicity_values,
                                      'Smart Periodicity': smart_periodicity_values})

    periodicity_df['Delta'] = abs(periodicity_df['Benign Periodicity'] - periodicity_df['Smart Periodicity'])

    periodicity_df = periodicity_df.sort_values(by='Delta', ascending=False)

    periodicity_df['ID'] = periodicity_df['ID'].apply(dec_to_hex)
    
    plot_periodicity_df(periodicity_df, os.path.join(save_loc, 'all_arb_ids.png'))
    
    top_5_delta_ids = periodicity_df.head(5)

    plot_periodicity_df(top_5_delta_ids, os.path.join(save_loc, 'top5_arb_ids.png'))
    
    
fname = 'exp_factor_' + str(factor)
out_dir = os.path.join(data_folder + fname)
os.makedirs(out_dir, exist_ok = True)


output_file = os.path.join(out_dir, 'smart_output.csv')
out_df.to_csv(output_file, index = False)

plot_outputs(normal_data, out_df, out_dir)



with open(os.path.join(out_dir,'data_desc.txt'),'a') as file:
    file.write(f"Factor: {factor}")
    file.write("\n")
    file.write(f"0: {len(out_df[out_df['label'] == 0])}")
    file.write("\n")
    file.write(f"1: {len(out_df[out_df['label'] == 1])}")




