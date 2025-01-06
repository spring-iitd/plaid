import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import pandas as pd
from sklearn.manifold import TSNE
import pickle
import joblib
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.utils as vutils
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from image_to_traffic import process_multiple_images
from traffic_preprocessing import process_traffic_logs
# Define transformations and dataset paths
data_transforms = {
        'test': transforms.Compose([transforms.ToTensor()]),
        'train': transforms.Compose([transforms.ToTensor()])
    }

# Define the MLP Model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)  # Apply softmax to the output

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

class InceptionResNetABlock(nn.Module):
    def __init__(self, in_channels, scale=0.17):
        super(InceptionResNetABlock, self).__init__()
        self.scale = scale
        self.branch0 = nn.Conv2d(in_channels, 32, kernel_size=1, stride=1, padding=0)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )
        self.conv_up = nn.Conv2d(96, in_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        mixed = torch.cat([branch0, branch1, branch2], dim=1)
        up = self.conv_up(mixed)
        return F.relu(x + self.scale * up)
    
# Inception-ResNet Model
class InceptionResNetV1(nn.Module):
    def __init__(self, num_classes=2):
        super(InceptionResNetV1, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.inception_a = InceptionResNetABlock(in_channels=128)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.inception_a(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
    

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the output from the last time step
        return out

def load_labels(label_file):
    """Load image labels from the label file."""
    labels = {}
    with open(label_file, 'r') as file:
        for line in file:
            # Ensure to strip extra characters like quotes and spaces
            filename, label = line.strip().replace("'", "").replace('"', '').split(': ')
            labels[filename.strip()] = int(label.strip())
    # print(type(labels))
    return labels

def load_dataset(data_dir,label_file,device):
    # Load datasets
    image_labels = load_labels(label_file)
    # Load images and create lists for images and labels
    images = []
    labels = []

    for filename, label in image_labels.items():
        img_path = os.path.join(data_dir, filename)
        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
            image = data_transforms['test'](image)  # Apply testing transformations
            images.append(image)
            labels.append(label)

    # Create tensors and send them to the specified device
    images_tensor = torch.stack(images).to(device)
    # labels_tensor = torch.tensor(labels)

    # Create DataLoader
    # dataset = TensorDataset(images_tensor, labels_tensor)
    # batch_size = 32 if is_train else 1  # Use larger batch size for training
    data_loader = DataLoader(images_tensor, batch_size=1, shuffle=False, num_workers=0)

    print(f'Loaded {len(images)} images.')
    return data_loader, labels

def load_test_model(surr_model_type,surrogate_model_path,test_model_type,test_model_path):
    # Define the number of classes in the dataset
    num_classes = 2
    
    
    surr_model_type == 'densenet161'
    model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    
    if test_model_type == 'densenet161':
        print("You have selected densenet161 as Target Model")
        test_model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
        test_model.classifier = nn.Linear(test_model.classifier.in_features, num_classes)
    elif test_model_type == 'densenet201':
        print("You have selected densenet201 as Target Model")
        test_model = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
        test_model.classifier = nn.Linear(test_model.classifier.in_features, num_classes)
    elif test_model_type == 'densenet169':
        print("You have selected densenet169 as Target Model")
        test_model = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT)
        test_model.classifier = nn.Linear(test_model.classifier.in_features, num_classes)
    elif test_model_type == 'densenet121':
        print("You have selected densenet121 as Target Model")
        test_model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        test_model.classifier = nn.Linear(test_model.classifier.in_features, num_classes)
    elif test_model_type == 'resnet18':
        print("You have selected Resnet18 as Target Model")
        test_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        test_model.fc = nn.Linear(test_model.fc.in_features, num_classes)
    elif test_model_type == 'convnext':
        print("You have selected Convnext as Target Model")
        test_model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        test_model.classifier[2] = nn.Linear(test_model.classifier[2].in_features, num_classes)
    elif test_model_type == 'wisa':
        print("You have selected Reduced Inception Resnet as Target Model")
        test_model = InceptionResNetV1(num_classes=2)
    elif test_model_type == 'MLP':
        print("You have selected MLP as Target Model")
        test_model = MLP(input_size=4, hidden_size=128, output_size=4)
    elif test_model_type == 'LSTM':
        print("You have selected LSTM as Target Model")
        test_model = LSTMModel(input_size=4, hidden_size=128, output_size=4)
    elif test_model_type == 'CNN':
        print("You have selected CNN as Target Model")
        pass
    
    

    # if surr_model_type == 'resnet18':
    #     # test_model = models.resnet18(pretrained=True)
    #     model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    #     model.fc = nn.Linear(model.fc.in_features, num_classes)
    # elif surr_model_type == 'densenet161':
    #     model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
    #     model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    # elif surr_model_type == 'densenet201':
    #     model = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
    #     model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    # elif surr_model_type == 'densenet121':
    #     model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    #     model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    # elif surr_model_type == 'densenet169':
    #     model = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT)
    #     model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    # else:
    #     model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
    #     model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    # if test_model_type == 'resnet18':
    #     # test_model = models.resnet18(pretrained=True)
    #     test_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    #     test_model.fc = nn.Linear(test_model.fc.in_features, num_classes)
    # elif test_model_type == 'resnet50':
    #     # test_model = models.resnet18(pretrained=True)
    #     test_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    #     test_model.fc = nn.Linear(test_model.fc.in_features, num_classes)
    # elif test_model_type == 'densenet161':
    #     test_model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
    #     test_model.classifier = nn.Linear(test_model.classifier.in_features, num_classes)
    # elif test_model_type == 'densenet201':
    #     test_model = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
    #     test_model.classifier = nn.Linear(test_model.classifier.in_features, num_classes)
    # elif test_model_type == 'densenet121':
    #     test_model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    #     test_model.classifier = nn.Linear(test_model.classifier.in_features, num_classes)
    # elif test_model_type == 'densenet169':
    #     test_model = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT)
    #     test_model.classifier = nn.Linear(test_model.classifier.in_features, num_classes)
    # else:
    #     test_model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
    #     test_model.classifier[2] = nn.Linear(test_model.classifier[2].in_features, num_classes)
    
    #If the systen don't have GPU
    # model.load_state_dict(torch.load(surrogate_model_path, map_location=torch.device('cpu')))
    # test_model.load_state_dict(torch.load(test_model_path, map_location=torch.device('cpu')))

    #If the system has GPU
    model.load_state_dict(torch.load(surrogate_model_path, weights_only=True))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if test_model_type == 'RF':
        print("You have selected RF as Target Model")
        test_model = joblib.load(test_model_path)
    else:
        test_model.load_state_dict(torch.load(test_model_path, weights_only=True))
        test_model = test_model.to(device)
        test_model.eval()

    
    
    model = model.to(device)

    model.eval()
    

    return model, test_model

def test_image_models(surrogate_model, test_model, test_loader,all_labels,target_model_type):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize lists to store predictions and labels
    all_surrogate_preds = []
    all_target_preds = []
    batch_indices = []
    # surrogate_features = []
    # target_features = []
    tsne_features = []
    tsne_labels = [] 
    label_idx = 0  
    surrogate_model.eval()
    test_model.eval()

    # Evaluate the model on the test dataset
    for batch_idx, inputs in enumerate(test_loader):
        inputs = inputs.to(device)  # Move data to the appropriate device
        with torch.no_grad():  # Disable gradient calculation for evaluation
            s_outputs = surrogate_model(inputs)  # Forward pass
            _, s_preds = torch.max(s_outputs, 1)  # Get predicted class labels
            t_outputs = test_model(inputs)  # Forward pass
            _, t_preds = torch.max(t_outputs, 1)  # Get predicted class labels
            
        # Store features for t-SNE visualization
        tsne_features.extend(t_outputs.cpu().numpy().reshape(len(t_outputs), -1))  # Flatten inputs
        
        ## Assign t-SNE labels based on predictions and all_labels
        for pred in t_preds.cpu().numpy():
            label = all_labels[label_idx]  # Get ground truth label from all_labels
            label_idx += 1  # Move to the next label
            # print("t_preds n label",pred, label)
            if pred == 0 and label == 0:
                tsne_labels.append("Original Benign")
            elif pred == 1 and label == 0:
                tsne_labels.append("False Positive")
            elif pred == 1 and label == 1:
                tsne_labels.append("Unsuccessful Perturbations")
            elif pred == 0 and label == 1:
                tsne_labels.append("Successful Perturbations")
            else:
                tsne_labels.append("Unknown Category")  # Optional fallback
        
        # # Classify the points (successful in surrogate and unsuccessful in target)
        # for idx, (pred, label, t_pred) in enumerate(zip(s_preds.cpu().numpy(), t_preds.cpu().numpy(), all_labels)):
        #     high_tsne_labels.append("Benign" if label == 0 else "Adversarial")

        #     # Highlight the points where it's successful in surrogate and unsuccessful in target
        #     if pred != label and t_pred == label:  # Misclassified by surrogate, correctly classified by target
        #         highlight_points.append(len(high_tsne_labels) - 1)  # Store the index for highlighting

        
        # Store predictions and labels
        all_surrogate_preds.extend(s_preds.cpu().numpy())
        all_target_preds.extend(t_preds.cpu().numpy())
        batch_indices.extend([batch_idx] * len(s_preds))  # Add the batch index for each prediction
        # all_labels.extend(labels.cpu().numpy())

    # Convert lists to numpy arrays
    all_surrogate_preds = np.array(all_surrogate_preds)
    all_target_preds = np.array(all_target_preds)
    # all_labels = np.array(all_labels)
    # Create a DataFrame to save predictions in a table
    data = {
        "Batch Index": batch_indices,
        "Surrogate Predictions": all_surrogate_preds,
        "Target Predictions": all_target_preds,
        "True labels": all_labels,
        "t-SNE Labels": tsne_labels
    }

    df = pd.DataFrame(data)
    output_file = "target_{}.csv".format(target_model_type)
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

    return all_surrogate_preds, all_target_preds,tsne_features, tsne_labels  

def extract_original_features(data_loader,all_labels, model, device,save_path='benign_features'):
    """
    Extract features from the original data using a trained model.

    Args:
        data_loader: DataLoader containing the original dataset.
        model: Trained model (PyTorch).
        device: Device to use for computation (CPU/GPU).

    Returns:
        features: List of extracted feature vectors.
        labels: List of corresponding labels (attack or benign).
    """
    model.eval()  # Set the model to evaluation mode
    features = []
    benign_features = []

    with torch.no_grad():
        for inputs in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)  # Pass through the model
            features.extend(outputs.cpu().numpy().reshape(len(outputs), -1))  # Extract features
            # labels.extend(targets.cpu().numpy())  # Get labels

    return np.array(features)

    # with torch.no_grad():
    #     for idx, inputs in enumerate(data_loader):
    #         inputs = inputs.to(device)
    #         outputs = model(inputs)  # Extract features
    #         batch_label = all_labels[idx]  # Retrieve the corresponding label
    #         batch_features = outputs.cpu().numpy()
    #         if batch_label == 0:
    #             benign_features.extend(batch_features.reshape(len(outputs), -1))

    # print("Benign features shape:", len(benign_features))
    # # Save the benign features
    # with open(save_path, 'wb') as f:
    #     pickle.dump(np.array(benign_features), f)
    # print(f"Benign features saved to {save_path}")

    ## return np.array(benign_features)

def plot_tsne_original(features, labels, save_path="original_input_data_tsne.png"):
    """
    Perform t-SNE dimensionality reduction and visualize the original data.

    Args:
        features: Array of feature vectors (original data).
        labels: Array of labels (attack: 1, benign: 0).
        save_path: File path to save the visualization plot.
    """
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(features)

    plt.figure(figsize=(10, 7))
    unique_labels = np.unique(labels)
    colors = {0: "blue", 1: "red"}  # Blue for benign, red for attack

    for label in unique_labels:
        indices = labels == label
        label_name = "Benign" if label == 0 else "Attack"
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=label_name, c=colors[label], alpha=0.6)

    plt.title("t-SNE Visualization of Original Data")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"t-SNE plot saved to {save_path}")

def evaluate_model(all_labels, all_preds):

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    # plt.savefig('./CF/perturb_attack_only/cf_{}_{}.png'.format(perturbation_type,max_perturbations), dpi=300)
    plt.show()
    

    # Now you can access the true negatives and other metrics
    true_negatives = cm[0, 0]
    false_positives = cm[0, 1]
    false_negatives = cm[1, 0]
    true_positives = cm[1, 1]

    # Calculate metrics
    tnr = true_negatives / (true_negatives + false_positives)  # True Negative Rate
    mdr = true_positives / (true_positives + false_negatives)  # malicious Detection Rate
    IDS_accu = accuracy_score(all_labels, all_preds) 
    IDS_prec = precision_score(all_labels, all_preds)
    IDS_recall = recall_score (all_labels,all_preds)
    IDS_F1 = f1_score(all_labels,all_preds)

    # Number of attack packets misclassified as benign (all_labels == 0 and all_preds == 1)
    # misclassified_attack_packets = ((all_labels == 1) & (all_preds == 0)).sum().item()
    misclassified_attack_packets = sum(1 for label, pred in zip(all_labels, all_preds) if label == 1 and pred == 0)
    #If label is a tensor
    # total_attack_packets = (all_labels == 1).sum().item()
    #if labels is a list
    total_attack_packets = 1361
    # total_attack_packets = 11834
    # attack_accuracy = sum(1 for label, pred in zip(all_labels, all_preds) if label == 1 and pred == 1) / total_attack_packets if total_attack_packets > 0 else 0


    oa_asr = misclassified_attack_packets / total_attack_packets

    return tnr, mdr, oa_asr, IDS_accu, IDS_prec, IDS_recall, IDS_F1

def compare_models(all_labels, all_surrogate_preds, all_target_preds):
    
    attack_indices = [i for i, label in enumerate(all_labels) if label == 1]
    # Use the attack_indices to filter the corresponding lists
    filtered_surrogate_preds = [all_surrogate_preds[i] for i in attack_indices]
    filtered_target_preds = [all_target_preds[i] for i in attack_indices]
    filtered_labels = [all_labels[i] for i in attack_indices]

    # Calculate the number of matching predictions between surrogate and target models
    # matching_predictions = (filtered_surrogate_preds == filtered_target_preds).sum()
    matching_predictions = sum(1 for s_pred, t_pred in zip(filtered_surrogate_preds, filtered_target_preds) if s_pred == t_pred)

    # print(type(filtered_labels))
    total_predictions = len(filtered_labels)

    # Calculate transferability rate
    transferability_rate = matching_predictions / total_predictions if total_predictions > 0 else 0

    print(f'\nTransferability Rate for attack images: {transferability_rate:.4f}')

def plot_tsne(tsne_features, tsne_labels, save_path):
    """
    Perform t-SNE dimensionality reduction and visualize the data.

    Args:
        tsne_features: List of feature vectors (input data for t-SNE).
        tsne_labels: List of labels corresponding to the features.
        perplexity: t-SNE perplexity value (controls clustering sensitivity).
        save_path: File path to save the visualization plot.
    """
    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=20)
    tsne_results = tsne.fit_transform(np.array(tsne_features))

    # Create a scatter plot for t-SNE results
    plt.figure(figsize=(10, 7))
    unique_labels = set(tsne_labels)
    colors = {
        "Original Benign": "blue",
        "Unsuccessful Perturbations": "red",
        "Successful Perturbations": "green",
        "Unknown Category": "gray"
    }

    for label in unique_labels:
        indices = [i for i, lbl in enumerate(tsne_labels) if lbl == label]
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=label, c=colors.get(label, "black"), alpha=0.6)

    plt.title("t-SNE Visualization of Perturbed and Benign Samples")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"t-SNE plot saved to {save_path}")   

def load_benign_features(save_path='benign_features'):
    """
    Load previously saved benign features.

    Args:
        save_path: Path to the saved benign features file.

    Returns:
        benign_features: Loaded benign features as a numpy array.
    """
    with open(save_path, 'rb') as f:
        benign_features = pickle.load(f)
    print(f"Benign features loaded from {save_path}")
    return benign_features

def shift_columns(df):
    for dlc in [2, 5, 6]:
        # Ensure compatibility by casting columns to a compatible type (object for mixed types)
        target_columns = df.columns[3:]
        df[target_columns] = df[target_columns].astype(object)

        # Perform the shift operation
        df.loc[df['dlc'] == dlc, target_columns] = (
            df.loc[df['dlc'] == dlc, target_columns]
            .shift(periods=8 - dlc, axis='columns', fill_value='00')
        )

    return df

def read_attack_data(data_path):

    columns = ['timestamp','can_id', 'dlc', 'data0', 'data1', 'data2', 'data3', 'data4',
           'data5', 'data6', 'data7', 'flag']

    data = pd.read_csv(data_path, names = columns,skiprows=1)
    data = shift_columns(data)
    ##Replacing all NaNs with '00'
    data = data.replace(np.NaN, '00')
    ##Joining all data columns to put all data in one column
    data_cols = ['data0', 'data1', 'data2', 'data3', 'data4', 'data5', 'data6', 'data7']
    ##The data column is in hexadecimal
    data['data'] = data[data_cols].apply(''.join, axis=1)
    data.drop(columns = data_cols, inplace = True, axis = 1)

    ##Converting columns to decimal
    data['can_id'] = data['can_id'].apply(hex_to_dec)
    data['data'] = data['data'].apply(hex_to_dec)

    data = data.assign(IAT=data['timestamp'].diff().fillna(0))

    if 'flag' in data.columns:
        # print(f"Before conversion, 'flag' column type: {data['flag'].dtypes}")
        data['flag'] = data['flag'].astype('int64')
        # print(f"After conversion, 'flag' column type: {data['flag'].dtypes}")

    return data

## Function to create a sequencified dataset for LSTM moodel
def sequencify(dataset, target, start, end=None, window=10):
    """
    Converts a dataset into a sequencified format suitable for sequence models like LSTM.

    Args:
        dataset (np.ndarray): The feature dataset (2D array).
        target (np.ndarray): The target labels corresponding to the dataset.
        start (int): The starting index for sequencing.
        end (int or None): The ending index for sequencing. If None, defaults to the length of the dataset.
        window (int): The window size for sequencing.

    Returns:
        np.ndarray: Sequencified features.
        np.ndarray: Corresponding target labels.
    """
    X = []
    y = []

    # Adjust start index to accommodate the window
    start = start + window
    if end is None:
        end = len(dataset)

    # Create sequences
    for i in range(start, end + 1):
        indices = range(i - window, i)
        X.append(dataset[indices])
        y.append(target[i - 1])  # Use the label corresponding to the last window index

    return np.array(X), np.array(y)

hex_to_dec = lambda x: int(x, 16)

def perturbed_traffic_pre_processing(mean, std,processed_output_file,target_model_type):
    # Assuming `read_attack_data` is defined elsewhere in your project
    # perturbed_dos_data_path = "/content/drive/MyDrive/HCRL_CH/converted_traffic_with_labels.csv"
    perturbed_dos_data = read_attack_data(processed_output_file)
    # print("perturbed DOS:", perturbed_dos_data['flag'].value_counts())

    # Extract features and labels
    X_dos_perturbed = perturbed_dos_data[['can_id', 'dlc', 'data', 'IAT']].values
    y_dos_perturbed = perturbed_dos_data['flag'].values

    # Load the scaler used during training
    scaler = joblib.load('scaler_m0.sav')
    X_dos_perturbed = scaler.transform(X_dos_perturbed)

    if target_model_type == "LSTM":
        # Sequencify the data
        X_test_seq, y_test_seq = sequencify(dataset=X_dos_perturbed, target=y_dos_perturbed, window=10, start=0, end=None)

        # Normalize the sequencified data using training mean and std
        X_test_seq -= mean  # Assume `mean` was saved from training time
        X_test_seq /= std   # Assume `std` was saved from training time
        print("Sequenced and normalized test data ready for LSTM model.")

        return X_test_seq, y_test_seq

    return X_dos_perturbed, y_dos_perturbed, 


    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)  # Apply softmax to the output

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

# Create DataLoaders
def create_test_loaders(X_test, y_test, batch_size=32):
    
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader
  
# Testing MLP model
def test_mlp_model(mlp,test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp = mlp.to(device)
    mlp.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = mlp(batch_X)
            predictions = torch.argmax(outputs, dim=1)
            y_true.extend(batch_y.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    print(f"ACCURACY: {accuracy}")
    print("CLASSIFICATION REPORT:\n", classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
    return y_true, y_pred

def test_rf_model(rf, X_test, y_test):
    rf_preds = rf.predict(X_test)

    print("-------RANDOM FOREST-------")
    accuracy = accuracy_score(y_test, rf_preds)
    print("ACCURACY: ", accuracy)
    print("CLASSIFICATION REPORT:\n", classification_report(y_test, rf_preds))

    cm = confusion_matrix(y_test, rf_preds)
    print("Confusion Matrix:\n", cm)
    return rf_preds, y_test

def test_lstm_model(test_model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_model = test_model.to(device)
    test_model.eval()
    
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = test_model(batch_X)
            predictions = torch.argmax(outputs, dim=1)
            y_true.extend(batch_y.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"ACCURACY: {accuracy}")
    print("CLASSIFICATION REPORT:\n", classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
    # print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    return y_true,y_pred

def test_packet_IDS(test_model,processed_output_file,target_model_type,mean,std,mutation_operation):
    X_dos_perturbed, y_dos_perturbed = perturbed_traffic_pre_processing(mean, std,processed_output_file,target_model_type)
    test_loader = create_test_loaders(X_dos_perturbed, y_dos_perturbed, batch_size=32)
    if target_model_type == "MLP":
        y_true,y_pred = test_mlp_model(test_model, test_loader)
    elif target_model_type == "RF":
        y_true,y_pred = test_rf_model(test_model, X_dos_perturbed, y_dos_perturbed)
    elif target_model_type == "LSTM":
        lstm_test_loader = create_test_loaders(X_dos_perturbed, y_dos_perturbed, batch_size=32)
        y_true,y_pred = test_lstm_model(test_model, lstm_test_loader)

    data = {
        "Predictions": y_pred,
        "True Labels": y_true,
    }

    df = pd.DataFrame(data)
    output_file = "prediction_results_{}_{}.csv".format(target_model_type,mutation_operation)
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

    return y_true,y_pred


def main():
    # Define the hyperparameters
    surr_model_type = 'densenet161'
    surrogate_model_path = "./Trained_Models/custom_cnn_model_chd_densenet_161.pth"

    if len(sys.argv) != 3:
        print("Usage: python file_name.py <Target Model Name>")
        sys.exit(1)

    # Read the perturbation type from the command-line argument
    target_model_type = sys.argv[1]
    mutation_operation = sys.argv[2]


    if target_model_type == 'densenet169':
        target_model_path = "./Trained_Models/custom_cnn_model_chd_densenet_169.pth"
    elif target_model_type == 'densenet201':
        target_model_path = "./Trained_Models/custom_cnn_model_chd_densenet_201.pth"
    elif target_model_type == 'resnet18':
        target_model_path = "./Trained_Models/custom_cnn_model_chd_resnet_18.pth"
    elif target_model_type == 'convnext':
        target_model_path = "./Trained_Models/custom_cnn_model_chd_convnext.pth"
    elif target_model_type == 'densenet121':
        target_model_path = "./Trained_Models/custom_cnn_model_chd_densenet_121.pth"
    elif target_model_type == 'wisa':
        target_model_path = "./Trained_Models/WISA_model.pth"
    elif target_model_type == 'MLP':
        target_model_path = "./Trained_Models/best_mlp_model.pth"
    elif target_model_type == 'RF':
        target_model_path = "./Trained_Models/best_rf_model.joblib"
    elif target_model_type == 'LSTM':
        target_model_path = "./Trained_Models/best_lstm_model.pth"
    elif target_model_type == 'CNN':
        target_model_path = "./Trained_Models/cnn_model.h5"
    else:
        print("Invalid target model type. Please choose from densenet169, densenet201, resnet18, convnext, MLP, RF, LSTM, CNN")
        sys.exit(1)
    
    if mutation_operation == 'Injection':
        test_dataset_dir = './Images_Injection' 
    elif mutation_operation == 'Modification':  
        test_dataset_dir = './Images_Modification'  
    elif mutation_operation == 'Both':
        test_dataset_dir = './Images_Inject_and_modify'
    
    
    test_label_file = "perturbed_images_true_labels.txt"
    orignal_traffic = 'original_traffic.txt'  # Update with the correct path if needed
    perturbed_traffic = f'perturbed_traffic_{mutation_operation}.txt'  # Update with the correct path if needed
    processed_output_file = f'converted_perturbed_traffic_{mutation_operation}.csv'  # Output Excel file
    
    # process_multiple_images("None",output_file = "original_traffic.txt")
    # print("--------------original traffic file created----------------")
    # process_traffic_logs("None",orignal_traffic, "converted_original_traffic.csv")

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = np.array([7.01463645e+02, 7.93270240e+00, 4.00606498e+18, 7.26353323e-04])
    std = np.array([4.24555666e+02, 5.88329352e-01, 6.46075121e+18, 1.38693241e-03])
    
    #laod the model
    surrogate_model, test_model = load_test_model(surr_model_type,surrogate_model_path,target_model_type,target_model_path)
    print("loaded surrogate and Target models")
    
    if target_model_type == "densenet169" or target_model_type == "densenet201" or target_model_type == "resnet18" or target_model_type == "convnext" or target_model_type == "densenet121" or target_model_type == "wisa":
        print("---------------Image based Target IDS-----------------")
        #load_dataset
        image_loader, all_labels = load_dataset(test_dataset_dir,test_label_file,device)
        # print(image_loader)
        print("loaded test image dataset")

        # benign_features = extract_original_features(image_loader,all_labels, surrogate_model, device)
        
        # original_features = extract_original_features(image_loader,all_labels, surrogate_model, device)
        # plot_tsne_original(original_features, all_labels, save_path="original_output_tsne_mg_201.png")

        print("---------------Image based IDS Performance-----------------")
        all_surrogate_preds, all_target_preds,tsne_features, tsne_labels = test_image_models(surrogate_model, test_model, image_loader,all_labels,target_model_type)
        # plot_tsne(tsne_features, tsne_labels, save_path="tsne_output_visualization_target121_surr161.png")
    
    else:
        print("---------------Packet based Target IDS-----------------")
        print("---------------Converting Image to traffic-----------------")
        # process_multiple_images(mutation_operation,output_file = f'perturbed_traffic_{mutation_operation}.txt')
        # print(f"perturbed traffic file created - perturbed_traffic_{mutation_operation}.txt")
        # process_traffic_logs(mutation_operation,perturbed_traffic, processed_output_file)
        # print("---------------Packet based IDS performance-----------------")
        all_labels, all_target_preds = test_packet_IDS(test_model,processed_output_file,target_model_type,mean,std,mutation_operation)
    
    

    # print("\nTarget Model Performance - {}".format(target_model_type))
    tnr, mdr, oa_asr, IDS_accu, IDS_prec, IDS_recall,IDS_F1 = evaluate_model(all_labels, all_target_preds)
    print("----------------IDS Perormance Metric----------------")
    print(f'Accuracy: {IDS_accu:.4f}')
    print(f'Precision: {IDS_prec:.4f}')
    print(f'Recall: {IDS_recall:.4f}')
    print(f'F1 Score: {IDS_F1:.4f}')

    print("----------------Adversarial attack Perormance Metric----------------")
    print("TNR:", tnr)
    print("Malcious Detection Rate:", mdr)
    print("Attack Success Rate:", oa_asr)


    # # compare_models(all_labels, all_surrogate_preds, all_target_preds)

    
if __name__ == "__main__":
    main()


