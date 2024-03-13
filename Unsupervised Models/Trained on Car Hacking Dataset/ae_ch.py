import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report, accuracy_score
from joblib import load
from tensorflow.keras.models import load_model
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
import tensorflow as tf
from tensorflow.distribute import MirroredStrategy
from joblib import dump

data_path = '../Smart Attack Algorithm/data/Car Hacking Dataset/benign_data.csv'

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
    data = data.drop(columns = ['Timestamp'], axis = 1)
    
    return data

normal_data = read_data(data_path)
normal_data.drop(columns = ['label'], inplace = True)

X_train, X_test = train_test_split(normal_data, test_size=0.2, random_state=42)

scaler = StandardScaler()

# scaler = load('scaler.joblib')
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

strat = MirroredStrategy()

EPOCHS = 1000
BATCH_SIZE = 32 * strat.num_replicas_in_sync
LOSS = 'mse'

# Define early stopping callback

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=20, verbose = 1)

early_stopper = EarlyStopping(monitor='val_loss', patience=60, restore_best_weights=True, verbose = 1)

input_dim = X_train.shape[1]


with strat.scope():
    model = Sequential()

    ##Encoder
    model.add(Dense(input_dim, input_shape=(input_dim, ), activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(2, activation='relu'))

    ##Bottleneck
    model.add(Dense(1, activation='relu'))

    ##Decoder
    model.add(Dense(2, activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(input_dim))
    
 

    model.compile(optimizer='adam', loss=LOSS)

history = model.fit(X_train, X_train, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                    validation_data=(X_test, X_test), callbacks=[reduce_lr, early_stopper])

model.save('ae_ch.h5')
dump(scaler, 'ch_scaler.joblib')

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')
plt.savefig('ch_ae_loss_curve.png')
plt.close()