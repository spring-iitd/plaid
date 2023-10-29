import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
import argparse
from joblib import dump
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()

parser.add_argument('directory', 
                    type=str, 
                    help = 'directory where files are stored')

parser.add_argument('file_name',
                    type = str,
                    help = 'dos traffic file')


args = parser.parse_args()


base_dir = args.directory
data_path = os.path.join(base_dir, args.file_name)
data = pd.read_csv(data_path)

data.drop(['Timestamp'], axis = 1, inplace=True)

data = data[:500_000]

def hex_to_bin(hex_num):
    
    binary_value = bin(int(str(hex_num), 16))[2:]
    
    return binary_value

def int_to_bin(int_num):
    
    binary_value = bin(int_num)[2:]
    
    return binary_value

def pad(value, length):
    
    curr_length = len(str(value))
    
    zeros = '0' * (length - curr_length)
    
    return zeros + value

hex_to_dec = lambda x: int(x, 16)

def transform_data(data):

    data['ID'] = data['ID'].apply(hex_to_dec)
    data['Payload'] = data['Payload'].apply(hex_to_dec)

    return data

def sequencify_data(X, y, seq_size=10):
    
    # Calculate the maximum index to be considered based on sequence size
    max_index = (len(X) // seq_size) * seq_size

    X_seq = []
    y_seq = []
    for i in range(0, max_index, seq_size):
        X_seq.append(X[i:i+seq_size])
        y_seq.append(1 if 1 in y[i:i+seq_size] else 0)

    return np.array(X_seq), np.array(y_seq)

data = transform_data(data)

X = data.drop('label', axis = 1)
y = data['label']

X_seq, y_seq = sequencify_data(X, y)

#Splitting into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True)
X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(X_seq, y_seq, test_size = 0.2, shuffle= True)

#Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

dump(scaler, os.path.join(base_dir, 'scaler.joblib'))

seq_scaler = StandardScaler()
num_train_samples, seq_length, num_features = X_seq_train.shape
num_test_samples, _, _ = X_seq_test.shape

X_train_seq_reshaped = X_seq_train.reshape(num_train_samples, -1)
X_test_seq_reshaped = X_seq_test.reshape(num_test_samples, -1)

X_train_seq_scaled = seq_scaler.fit_transform(X_train_seq_reshaped)
X_test_seq_scaled = seq_scaler.transform(X_test_seq_reshaped)

dump(scaler, os.path.join(base_dir, 'seq_scaler.joblib'))

# Reshape the scaled data back to the original shape
X_seq_train = X_train_seq_scaled.reshape(num_train_samples, seq_length, num_features)
X_seq_test = X_test_seq_scaled.reshape(num_test_samples, seq_length, num_features)

oversample = SMOTE()
X_train_smote, y_train_smote = oversample.fit_resample(X_train, y_train) 

##Models

print("-----MLP-------")

mlp = Sequential()
mlp.add(Input(shape = (4)))
mlp.add(Dense(128, activation = 'relu'))
mlp.add(Dense(128, activation = 'relu'))
mlp.add(Dense(1, activation = 'sigmoid'))

mlp.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

es = EarlyStopping(monitor = 'val_loss', patience = 15, restore_best_weights = True)

mlp_hist = mlp.fit(X_train_smote, y_train_smote, epochs=100, callbacks = [es], validation_split=0.2, batch_size = 128)

##MLP
print("-----MLP-------")

threshold = 0.5
mlp_preds = mlp.predict(X_test)
mlp_preds = (mlp_preds >= threshold).astype(int)

print("ACCURACY: ", accuracy_score(y_test, mlp_preds))
print("CLASSIFICATION REPORT:\n", classification_report(y_test, mlp_preds))

with open(os.path.join(base_dir,'evaluation_results.txt'),'w') as file:
    file.write("-------MLP-------\n")
    file.write(f"Accuracy Score: ")
    file.write(str(accuracy_score(y_test, mlp_preds)))
    file.write("\n")
    file.write('Classification Report:\n')
    file.write(str(classification_report(y_test, mlp_preds)))
    file.write("\n\n\n\n")

mlp.save(os.path.join(base_dir, 'mlp.h5'))

plt.figure(figsize=(10, 10))
plt.plot(mlp_hist.history['loss'])
plt.plot(mlp_hist.history['val_loss'])
plt.title('MLP Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(os.path.join(base_dir,'mlp_training_history.png'))

##LSTM

# print("-----LSTM-------")

# lstm = Sequential()

# lstm.add(Input(shape = X_seq_train.shape[1:]))
# lstm.add(LSTM(128, activation = 'relu'))
# lstm.add(Dense(1, activation = 'sigmoid'))

# lstm.compile(
#     loss = 'binary_crossentropy',
#     optimizer = 'adam',
#     metrics = ['accuracy'])

# es = EarlyStopping(monitor = 'val_loss', patience = 15, restore_best_weights = True)

# lstm_hist = lstm.fit(X_seq_train, y_seq_train, batch_size = 64, validation_split = 0.2,
#         callbacks = [es], epochs = 1000)

# print("-----LSTM-------")

# lstm_preds = lstm.predict(X_seq_test, batch_size=4096)
# lstm_preds = (lstm_preds >= threshold).astype(int)

# print("ACCURACY: ", accuracy_score(y_seq_test, lstm_preds))
# print("CLASSIFICATION REPORT:\n", classification_report(y_seq_test, lstm_preds))

# with open(os.path.join(base_dir,'evaluation_results.txt'),'w') as file:
#     file.write("-------LSTM-------\n")
#     file.write(f"Accuracy Score: ")
#      file.write(str(accuracy_score(y_test, lstm_preds)))
#      file.write("\n")
#     file.write('Classification Report:\n')
#     file.write(str(classification_report(y_test, lstm_preds)))
#     file.write("\n\n\n\n")

# lstm.save(os.path.join(base_dir, 'lstm.h5'))

# plt.figure(figsize=(10, 10))
# plt.plot(lstm_hist.history['loss'])
# plt.plot(lstm_hist.history['val_loss'])
# plt.title('LSTM Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.savefig(os.path.join(base_dir,'mlp_training_history.png'))


## XGBOOST
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)

print("-------XGBOOST-------")
print("ACCURACY: ", accuracy_score(y_test, xgb_preds))
print("CLASSIFICATION REPORT:\n", classification_report(y_test, xgb_preds))
xgb.save_model(os.path.join(base_dir, 'xgb.json'))

with open(os.path.join(base_dir,'evaluation_results.txt'),'a') as file:
    file.write("-------XGB-------\n")
    file.write(f"Accuracy Score: ")
    file.write(str(accuracy_score(y_test, xgb_preds)))
    file.write("\n")
    file.write('Classification Report:\n')
    file.write(str(classification_report(y_test, xgb_preds)))
    file.write("\n\n\n\n")

## DECISION TREE
dt = DecisionTreeClassifier(max_depth = 4)
dt.fit(X_train, y_train)
dt_preds = dt.predict(X_test)

print("-------DECISION TREE--------\n")
print("ACCURACY: ", accuracy_score(y_test, dt_preds))
print("CLASSIFICATION REPORT:\n", classification_report(y_test, dt_preds))
dump(dt, os.path.join(base_dir, 'dt.pkl'))

with open(os.path.join(base_dir,'evaluation_results.txt'),'a') as file:
    file.write("-------Decision Tree-------")
    file.write(f"Accuracy Score: ")
    file.write(str(accuracy_score(y_test, dt_preds)))
    file.write("\n")
    file.write('Classification Report:\n')
    file.write(str(classification_report(y_test, dt_preds)))
    file.write("\n\n\n\n")

## RANDOM FOREST

rf = RandomForestClassifier(n_estimators=100, max_depth=4)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

print("-------RANDOM FOREST-------\n")
print("ACCURACY: ", accuracy_score(y_test, rf_preds))
print("CLASSIFICATION REPORT:\n", classification_report(y_test, rf_preds))
dump(rf, os.path.join(base_dir, 'rf.pkl'))

with open(os.path.join(base_dir,'evaluation_results.txt'),'a') as file:
    file.write("-------Random Forest-------")
    file.write(f"Accuracy Score: ")
    file.write(str(accuracy_score(y_test, rf_preds)))
    file.write("\n")
    file.write('Classification Report:\n')
    file.write(str(classification_report(y_test, rf_preds)))
    file.write("\n\n\n\n")