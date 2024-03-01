import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from joblib import dump

data_path = "fixeddelta020take2.log"

def hex_to_bin(hex_num):
    
    binary_value = bin(int(str(hex_num), 16))[2:]
    
    return binary_value

def int_to_bin(int_num):
    
    binary_value = bin(int_num)[2:]
    
    return binary_value

hex_to_dec = lambda x: int(x, 16)
dec_to_hex = lambda x : hex(int(x))[2:]

columns = ['Timestamp', 'ID', 'DLC', 'Payload']

# Read the file into a list of lines
with open(data_path, 'r') as file:
    lines = file.readlines()

# Parse each line and extract the relevant information
data = []
for line in lines:
    parts = line.strip().split()
    timestamp = float(parts[0][1:-1])  # Remove parentheses
    bus_name = parts[1]
    ID = parts[2]
    DLC = int(parts[3][1:-1])  # Remove brackets and convert to integer
    payload = ''.join(parts[4:])  # Concatenate payload
    data.append([timestamp, ID, DLC, payload])

# Create a DataFrame from the parsed data
df = pd.DataFrame(data, columns=columns)

df['ID'] = df['ID'].apply(hex_to_dec)
df['Payload'] = df['Payload'].apply(hex_to_dec)
df = df.assign(IAT=df['Timestamp'].diff().fillna(0))

df.drop(columns=['Timestamp'], inplace=True)

X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

dump(scaler, 'scaler.joblib')

input_dim = X_train.shape[1]

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

EPOCHS = 1000
BATCH_SIZE = 32
LOSS = 'mse'

def lr_scheduler(epoch, lr):
    if epoch > 0 and epoch % 20 == 0:
        return lr / 10
    return lr

lr_callback = LearningRateScheduler(lr_scheduler, verbose=1)

# Define early stopping callback
early_stopper = EarlyStopping(monitor='val_loss', patience=60, restore_best_weights=True)

model.compile(optimizer='adam', loss=LOSS)

history = model.fit(X_train, X_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, X_test), callbacks=[lr_callback, early_stopper])


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('ae_loss_chart.png')
model.save('ae.h5')