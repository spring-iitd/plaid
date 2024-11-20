import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.utils as vutils
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Define transformations and dataset paths
data_transforms = {
        'test': transforms.Compose([transforms.ToTensor()]),
        'train': transforms.Compose([transforms.ToTensor()])
    }

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

def load_test_model(surrogate_model_path,test_model_path, test_model_type,surr_model_type):
    # labels = image_datasets.tensors[1]
    # unique_classes = torch.unique(labels)
    # num_classes = len(unique_classes)
    num_classes = 2
    
    if surr_model_type == 'resnet':
        # test_model = models.resnet18(pretrained=True)
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif surr_model_type == 'densenet':
        model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    if test_model_type == 'resnet':
        # test_model = models.resnet18(pretrained=True)
        test_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        test_model.fc = nn.Linear(test_model.fc.in_features, num_classes)
    elif test_model_type == 'densenet':
        test_model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
        test_model.classifier = nn.Linear(test_model.classifier.in_features, num_classes)
    else:
        test_model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        test_model.classifier[2] = nn.Linear(test_model.classifier[2].in_features, num_classes)
    
    #If the systen don't have GPU
    # model.load_state_dict(torch.load(surrogate_model_path, map_location=torch.device('cpu')))
    # test_model.load_state_dict(torch.load(test_model_path, map_location=torch.device('cpu')))

    #If the system has GPU
    model.load_state_dict(torch.load(surrogate_model_path, weights_only=True))
    test_model.load_state_dict(torch.load(test_model_path, weights_only=True))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_model = test_model.to(device)
    model = model.to(device)

    # model.eval()
    # test_model.eval()

    return model, test_model

def test_models(surrogate_model, test_model, test_loader):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize lists to store predictions and labels
    all_surrogate_preds = []
    all_target_preds = []

    surrogate_model.eval()
    test_model.eval()

    # Evaluate the model on the test dataset
    for inputs in test_loader:
        inputs = inputs.to(device)  # Move data to the appropriate device
        with torch.no_grad():  # Disable gradient calculation for evaluation
            s_outputs = surrogate_model(inputs)  # Forward pass
            _, s_preds = torch.max(s_outputs, 1)  # Get predicted class labels
            t_outputs = test_model(inputs)  # Forward pass
            _, t_preds = torch.max(t_outputs, 1)  # Get predicted class labels
        
        # Store predictions and labels
        all_surrogate_preds.extend(s_preds.cpu().numpy())
        all_target_preds.extend(t_preds.cpu().numpy())
        # all_labels.extend(labels.cpu().numpy())

    # Convert lists to numpy arrays
    all_surrogate_preds = np.array(all_surrogate_preds)
    all_target_preds = np.array(all_target_preds)
    # all_labels = np.array(all_labels)


    return all_surrogate_preds, all_target_preds  # Return accuracy for potential further use

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
    attack_accuracy = sum(1 for label, pred in zip(all_labels, all_preds) if label == 1 and pred == 1) / total_attack_packets if total_attack_packets > 0 else 0


    oa_asr = misclassified_attack_packets / total_attack_packets

    return tnr, mdr, oa_asr, IDS_accu, IDS_prec, IDS_recall, IDS_F1,attack_accuracy

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
    

def main():
    # Define the hyperparameters
    target_model_type = 'resnet'
    surr_model_type = 'resnet'

    # Define the dataset paths
    # test_dataset_dir = './test_random_densefeedback_20inj'
    # test_dataset_dir = './test_random_resnet_feedback_1inj' 
    test_dataset_dir = './test_random_resnet_feedback_20inj'
    # target_model_path = "./Trained_Models/custom_cnn_model_chd_convnext_.pth"
    # target_model_path = "./Trained_Models/custom_cnn_model_chd_densenet_.pth"
    target_model_path = "./Trained_Models/custom_cnn_model_chd_resnet_ 1.pth"
    test_label_file = "perturbed_images_true_labels.txt"
    # surrogate_model_path = "./Trained_Models/custom_cnn_model_chd_densenet_.pth"
    surrogate_model_path = "./Trained_Models/custom_cnn_model_chd_resnet_ 1.pth"

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #load_dataset
    image_loader, all_labels = load_dataset(test_dataset_dir,test_label_file,device)
    # print(type(all_labels))
    print("loaded test dataset")

    #laod the model
    surrogate_model, test_model = load_test_model(surrogate_model_path,target_model_path, target_model_type,surr_model_type)
    print("loaded surrogate model")
    all_surrogate_preds, all_target_preds = test_models(surrogate_model, test_model, image_loader)

    print("---------------Image based IDS performance-----------------")

    print("\nSurrogate Model Performance - {}".format(surr_model_type))
    tnr, mdr, oa_asr, IDS_accu, IDS_prec, IDS_recall,IDS_F1,attack_accuracy = evaluate_model(all_labels, all_surrogate_preds)
    print(f'Accuracy: {IDS_accu:.4f}')
    print(f'Precision: {IDS_prec:.4f}')
    print(f'Recall: {IDS_recall:.4f}')
    print(f'F1 Score: {IDS_F1:.4f}')
    print("TNR:", tnr)
    print("MDR:", mdr)
    print("OA_ASR:", oa_asr)
    print("Attack Accuracy:", attack_accuracy)
    

    print("\nTarget Model Performance - {}".format(target_model_type))
    tnr, mdr, oa_asr, IDS_accu, IDS_prec, IDS_recall,IDS_F1,attack_accuracy = evaluate_model(all_labels, all_target_preds)
    print(f'Accuracy: {IDS_accu:.4f}')
    print(f'Precision: {IDS_prec:.4f}')
    print(f'Recall: {IDS_recall:.4f}')
    print(f'F1 Score: {IDS_F1:.4f}')
    print("TNR:", tnr)
    print("MDR:", mdr)
    print("OA_ASR:", oa_asr)
    print("Attack Accuracy:", attack_accuracy)


    compare_models(all_labels, all_surrogate_preds, all_target_preds)

    # print("---------------Packet based IDS performance-----------------")
    # image_to_traffic()
    # evaluate_traffic_models()
    
if __name__ == "__main__":
    main()


