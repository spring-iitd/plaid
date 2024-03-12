import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization, RepeatVector, LeakyReLU, Flatten, TimeDistributed, Add, Conv1D, Concatenate, Lambda
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import time
from datetime import datetime
import joblib


print(tf.config.list_physical_devices('GPU'))


data_dir = '../car_hacking_data/'
os.listdir(data_dir)

benign_data_path = os.path.join(data_dir, "normal_run_data.txt")


hex_to_dec = lambda x: int(x, 16)

## Since there are varying DLCs (2,5,6,8) in order to maintain data integrity
## The data must be padded with 00s when DLC < 8

def shift_columns(df):
    
    for dlc in [2,5,6]:

        df.loc[df['dlc'] == dlc, df.columns[3:]] = df.loc[df['dlc'] == dlc, df.columns[3:]].shift(periods=8-dlc, axis='columns', fill_value='00')

    return df

def pad_with_zeros(string, desired_length=16):
    if len(string) >= desired_length:
        return string
    else:
        return string.zfill(desired_length)
    
def split_string_into_list(string):
    # Initialize an empty list to store the result
    result_list = []

    # Iterate through the string with a step size of 2
    for i in range(0, len(string), 2):
        # Extract two characters at a time and add them to the result list
        item = string[i:i+2]
        result_list.append(item)

    return result_list


def read_attack_data(data_path):
    
    columns = ['timestamp','can_id', 'dlc', 'data0', 'data1', 'data2', 'data3', 'data4', 
           'data5', 'data6', 'data7', 'flag']
    
    data = pd.read_csv(data_path, names = columns)

    data = shift_columns(data)
    
    ##Replacing all NaNs with '00' 
    data = data.replace(np.NaN, '00')
    
    ##Joining all data columns to put all data in one column
    data_cols = ['data0', 'data1', 'data2', 'data3', 'data4', 'data5', 'data6', 'data7']
    
    ##The data column is in hexadecimal
#     data['data'] = data[data_cols].apply(''.join, axis=1)
#     data.drop(columns = data_cols, inplace = True, axis = 1)
    
    ##Converting columns to decimal
    data['can_id'] = data['can_id'].apply(hex_to_dec)
    data[data_cols] = data[data_cols].astype(str)
    
    data.sort_values(by = ['timestamp'], inplace = True)
    data = data.assign(IAT=data['timestamp'].diff().fillna(0))
    data.drop(['timestamp'], inplace = True, axis = 1)
    
    data[data_cols] = data[data_cols].applymap(hex_to_dec)
    

    return data

    
  
timestamps = []
ids = []
dlcs = []
data = []
data_cols = ['data0', 'data1', 'data2', 'data3', 'data4', 'data5', 'data6', 'data7']
    
# Read the data from the file
with open(benign_data_path, 'r') as file:
    for line in file:
        # Extract information from each line
        line = line.strip()
        ts = line.split('Timestamp: ')[1].split(' ')[0]
        can_id = line.split('ID: ')[1].split(' ')[0]
        dlc = line.split('DLC: ')[1].split(' ')[0]
        can_data = ''.join(line.split('DLC: ')[1].split(' ')[1:])
        
        can_data = pad_with_zeros(can_data)
        data_split = split_string_into_list(can_data)
               
        #Converting Hexadecimal entries to decimal format
        timestamps.append(float(ts))
        ids.append(hex_to_dec(can_id))
        dlcs.append(int(dlc))
        data.append([hex_to_dec(hex_str) for hex_str in data_split])


    
        
# data_dict = {f"data{i}": col for i, col in enumerate(data_split)}
        
benign = pd.DataFrame({
    'timestamp': timestamps,
    'can_id': ids,
    'dlc': dlcs})

data = pd.DataFrame(data, columns = data_cols)

benign_data = pd.concat([benign, data], axis=1)
benign_data.sort_values(by = ['timestamp'], inplace = True)

# # Creating IAT column
benign_data= benign_data.assign(IAT=benign_data['timestamp'].diff().fillna(0))
benign_data.drop(columns = ['timestamp'], axis = 1, inplace= True)



X = benign_data.values

# test = read_attack_data(dos_data_path)
# x_test = test.drop(['flag'], axis = 1)
# y_test = test['flag'].replace({'R' : 0, 'T' : 1})

# x_test = x_test.values

val_idx = int(0.8 * len(X))

scaler = StandardScaler()
X_train = X[:val_idx]
X_val = X[val_idx:]

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

scaler_filename = "second_run/scaler.sav"
joblib.dump(scaler, scaler_filename) 
# X_test = scaler.transform(x_test)




## Function to create a sequencified dataset for time-series moodel
def sequencify(dataset, start, end, window):
  
    X = []
    
    start = start + window 
    if end is None:
        end = len(dataset)
        
    for i in range(start, end+1):
        indices = range(i-window, i) 
        X.append(dataset[indices])
			
    return np.array(X)


seq_size = 10

X_train_seq = sequencify(X_train, 0, None, seq_size)
X_val_seq = sequencify(X_val, 0, None, seq_size)


def TransformerBlock(inputs, num_heads, key_dim, ff_dim, dropout=0.3):
    multihead_attention = MultiHeadAttention(key_dim=key_dim, num_heads=num_heads)
    attention_output = multihead_attention(inputs, inputs, inputs)
    attention_output = Dropout(dropout)(attention_output)
    x = Add()([inputs, attention_output])
    x = LayerNormalization(epsilon=1e-6)(x)

    ffn_output = Dense(ff_dim, activation='relu')(x)
    ffn_output = Dense(x.shape[-1])(ffn_output)
    ffn_output = Dropout(dropout)(ffn_output)
    x = Add()([x, ffn_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x


## Change loss fn, figure out issue related to shape

def make_AE(latent_dim = 3, input_shape = (seq_size, 11), num_heads = 8, key_dim = 64, num_blocks = 8, 
            seq_size = seq_size):
    
    features = input_shape[-1]
    
    inp = Input(shape = input_shape, name = 'encoder_inp')
    
    # Create the MultiHeadAttention layer
    x = inp
    
    
    for _ in range(num_blocks):
        x = TransformerBlock(x, num_heads = num_heads, key_dim = key_dim, ff_dim = 16)
        
#     x = TimeDistributed(Dense((256), name = 'encoder_dense_3'))(x)
#     x = LeakyReLU(alpha = 0.2)(x)
    
#     x = TimeDistributed(Dense((128), name = 'encoder_dense_4'))(x)
#     x = LeakyReLU(alpha = 0.2)(x)
    
    x = TimeDistributed(Dense(features - 2), name = 'encoder_dense_1')(x)
    x = LeakyReLU(alpha = 0.2)(x)
   
    x1 = TimeDistributed(Dense(features - 4), name = 'encoder_dense_2')(x)
    x1 = LeakyReLU(alpha = 0.2)(x1)
    
    x2 = TimeDistributed(Dense(features - 6), name = 'encoder_dense_3')(x1)
    x2 = LeakyReLU(alpha = 0.2)(x2)
    
    x3 = TimeDistributed(Dense(features - 7), name = 'encoder_dense_4')(x2)
    x3 = LeakyReLU(alpha = 0.2)(x3)
    
    flattened_output = Flatten(name = 'encoder_flatten')(x3)
    
#     z_mean = layers.Dense(latent_dim, name="z_mean")(flattened_output)
#     z_log_var = layers.Dense(latent_dim, name="z_log_var")(flattened_output)
    
#     z = Sampling()([z_mean, z_log_var])

    code_layer = Dense(latent_dim, name = 'code')(flattened_output)
    
    encoder_ae = Model(inputs=inp,
                        outputs=code_layer,
                        name='Attention_AE_encoder')
    
    inp_decoder = Input(shape = (latent_dim,), name = 'decoder_inp')
    
    repeat_vec = RepeatVector(seq_size, name = 'repeat_vec')(inp_decoder)
    
    y = repeat_vec
    
    for _ in range(num_blocks):
        y = TransformerBlock(y, num_heads, key_dim, ff_dim = 16)
    
    y = TimeDistributed(Dense(features - 7), name = 'decoder_dense_1')(y)
    y = LeakyReLU(alpha = 0.2)(y)
    
    y1 = TimeDistributed(Dense(features - 6), name = 'decoder_dense_2')(y)
    y1 = LeakyReLU(alpha = 0.2)(y1)
    
    y2 = TimeDistributed(Dense(features - 4), name = 'decoder_dense_3')(y1)
    y2 = LeakyReLU(alpha = 0.2)(y2)
    
    y3 = TimeDistributed(Dense(features - 2), name = 'decoder_dense_4')(y2)
    y3 = LeakyReLU(alpha = 0.2)(y3)
    
    # Output layer
    output = TimeDistributed(Dense(input_shape[-1], activation='linear', name = 'decoder_op'))(y3)
    
    decoder_ae = Model(inputs=inp_decoder, outputs=output, name='Attention_AE_decoder')

    # AE model
    ae_inputs = inp
    z = encoder_ae(ae_inputs)
    ae_outputs = decoder_ae(z)
    ae = Model(inputs=ae_inputs, outputs=ae_outputs, name='Attention_AE')

    return encoder_ae, decoder_ae, ae 



encoder_ae, decoder_ae, ae = make_AE()


strat = tf.distribute.MirroredStrategy()

with strat.scope():
    encoder_ae, decoder_ae, ae = make_AE()
    ae.compile(loss =  'mae', optimizer = 'adam')
    
    
  

timestamp = time.time()
datetime_obj = datetime.fromtimestamp(timestamp)
fmt_time = datetime_obj.strftime('%m-%d %H:%M:%S')

tb = TensorBoard(log_dir=f'third_run/vae_logs/{fmt_time}')

es = EarlyStopping(monitor = 'val_loss', patience = 20, restore_best_weights=True)

ckpt = ModelCheckpoint(filepath = 'third_run/vae_cpkts/model-{epoch:02d}-{val_loss:.4f}.hdf5',
                      monitor = 'val_loss',
                      mode = 'min',
                      save_best_only = True,
                      verbose = 1)

red_lr = ReduceLROnPlateau(patience = 5, verbose = 1)

BATCH_SIZE = 256 * strat.num_replicas_in_sync


ae.fit(X_train_seq, X_train_seq, validation_data = (X_val_seq, X_val_seq), callbacks = [tb, es, ckpt, red_lr], 
         epochs = 10000)


ae.save('third_run/ae.h5')

