import os
import numpy as np
import pandas as pd
import argparse
from tensorflow.keras.models import load_model
from xgboost import XGBClassifier
from joblib import load
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

parser = argparse.ArgumentParser()

parser.add_argument('directory', 
                    type=str, 
                    help = 'directory where files are stored')

args = parser.parse_args()

data_dir = args.directory
file_name = 'smart_output.csv'
smart_data = pd.read_csv(os.path.join(data_dir, file_name))

smart_data = smart_data.assign(IAT=smart_data['Timestamp'].diff().fillna(0))
smart_data.drop(['Timestamp'], axis = 1, inplace = True)
smart_data.head()

X = smart_data.drop(['label'], axis = 1).to_numpy()
y = smart_data['label'].to_numpy()

def sequencify_data(X, y, seq_size=10):
    
    # Calculate the maximum index to be considered based on sequence size
    max_index = (len(X) // seq_size) * seq_size

    X_seq = []
    y_seq = []
    for i in range(0, max_index, seq_size):
        X_seq.append(X[i:i+seq_size])
        y_seq.append(1 if 1 in y[i:i+seq_size] else 0)

    return np.array(X_seq), np.array(y_seq)


scaler = load(os.path.join(data_dir, 'scaler.joblib'))
seq_scaler = load(os.path.join(data_dir, 'seq_scaler.joblib'))

mlp = load_model(os.path.join(data_dir, 'mlp.h5'))
# lstm = load_model(os.paht.join(data_dir, 'lstm.h5'))
xgb = XGBClassifier()
xgb.load_model(os.path.join(data_dir, 'xgb.json'))
dt = load(os.path.join(data_dir, 'dt.pkl'))
rf = load(os.path.join(data_dir, 'rf.pkl'))

X_seq, y_seq = sequencify_data(X, y)

X = scaler.transform(X)

num_samples, seq_length, num_features = X_seq.shape
X_seq_reshaped = X_seq.reshape(num_samples, -1)
X_seq = seq_scaler.fit_transform(X_seq_reshaped)
X_seq = X_seq.reshape(num_samples, seq_length, num_features)


threshold = 0.5

print("------MLP------")

mlp_preds = mlp.predict(X)
mlp_preds = (mlp_preds >= threshold).astype(int)

print("ACCURACY: ", accuracy_score(y, mlp_preds))
print("CLASSIFICATION REPORT:\n", classification_report(y, mlp_preds))

with open(os.path.join(data_dir,'evaluation_results.txt'),'w') as file:
    file.write("-------MLP-------\n")
    file.write(f"Accuracy Score: ")
    file.write(str(accuracy_score(y, mlp_preds)))
    file.write("\n")
    file.write('Classification Report:\n')
    file.write(str(classification_report(y, mlp_preds)))
    file.write("\n\n\n\n")


# print("------LSTM------")

# lstm_preds = lstm.predict(X_seq)
# lstm_preds = (lstm_preds >= threshold).astype(int)

# print("ACCURACY: ", accuracy_score(y_seq, lstm_preds))
# print("CLASSIFICATION REPORT:\n", classification_report(y_seq, lstm_preds))

# with open(os.path.join(data_dir,'evaluation_results.txt'),'a') as file:
#     file.write("-------LSTM-------\n")
#     file.write(f"Accuracy Score: ")
#     file.write(str(accuracy_score(y_seq, lstm_preds)))
#     file.write("\n")
#     file.write('Classification Report:\n')
#     file.write(str(classification_report(y_seq, lstm_preds)))
#     file.write("\n\n\n\n")

print("------XGBOOST------")

xgb_preds = xgb.predict(X)
print("ACCURACY: ", accuracy_score(y, xgb_preds))
print("CLASSIFICATION REPORT:\n", classification_report(y, xgb_preds))

with open(os.path.join(data_dir,'evaluation_results.txt'),'a') as file:
    file.write("-------XGBOOST-------\n")
    file.write(f"Accuracy Score: ")
    file.write(str(accuracy_score(y, xgb_preds)))
    file.write("\n")
    file.write('Classification Report:\n')
    file.write(str(classification_report(y, xgb_preds)))
    file.write("\n\n\n\n")

print("------Decision Tree------")

dt_preds = dt.predict(X)
print("ACCURACY: ", accuracy_score(y, dt_preds))
print("CLASSIFICATION REPORT:\n", classification_report(y, dt_preds))

with open(os.path.join(data_dir,'evaluation_results.txt'),'a') as file:
    file.write("-------Decision Tree-------\n")
    file.write(f"Accuracy Score: ")
    file.write(str(accuracy_score(y, dt_preds)))
    file.write("\n")
    file.write('Classification Report:\n')
    file.write(str(classification_report(y, dt_preds)))
    file.write("\n\n\n\n")

print("------Random Forest------")

rf_preds = rf.predict(X)
print("ACCURACY: ", accuracy_score(y, rf_preds))
print("CLASSIFICATION REPORT:\n", classification_report(y, rf_preds))

with open(os.path.join(data_dir,'evaluation_results.txt'),'a') as file:
    file.write("-------Random Forest-------\n")
    file.write(f"Accuracy Score: ")
    file.write(str(accuracy_score(y, rf_preds)))
    file.write("\n")
    file.write('Classification Report:\n')
    file.write(str(classification_report(y, rf_preds)))
    file.write("\n\n\n\n")