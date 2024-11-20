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
from sklearn.metrics import roc_curve, auc
import torchvision.utils as vutils
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision.utils import save_image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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
    # print(labels)
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
    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels)

    # Create DataLoader
    dataset = TensorDataset(images_tensor, labels_tensor)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f'Loaded {len(images)} images.')
    print(f'First image shape: {data_loader.dataset[0][0].shape}, Label: {data_loader.dataset[0][1]}')

    return dataset, data_loader

def load_model(pre_trained_model_path,test_model_path, test_model_type):
    # Load the pre-trained ResNet-18 model
    
    # labels = image_datasets.tensors[1]
    # unique_classes = torch.unique(labels)
    # num_classes = len(unique_classes)
    num_classes = 2
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if test_model_type == 'res':
        # test_model = models.resnet18(pretrained=True)
        test_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        test_model.fc = nn.Linear(test_model.fc.in_features, num_classes)
    elif test_model_type == 'dense':
        test_model = models.densenet161(pretrained=True)
        test_model.classifier = nn.Linear(test_model.classifier.in_features, num_classes)
    else:
        test_model = models.convnext_base(pretrained=True)
        test_model.classifier[2] = nn.Linear(test_model.classifier[2].in_features, num_classes)
    
    #If the systen don't have GPU
    # model.load_state_dict(torch.load(pre_trained_model_path, map_location=torch.device('cpu')))
    # test_model.load_state_dict(torch.load(test_model_path, map_location=torch.device('cpu')))

    #If the system has GPU
    model.load_state_dict(torch.load(pre_trained_model_path))
    test_model.load_state_dict(torch.load(test_model_path))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    test_model = test_model.to(device)
    
    model.eval()
    test_model.eval()

    return model, test_model

def evaluate_threshold_on_normal_data(model, test_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.eval()  # Set model to evaluation mode
    correct_probs = []
    incorrect_probs = []
    correct_indices = []  # To store indices of correct predictions
    incorrect_indices = []  # To store indices of incorrect predictions

    with torch.no_grad():
        for n_image, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred_probs = torch.softmax(output, dim=1)
            pred_class = output.max(1)[1]  # Get the predicted class
            pred_prob = pred_probs[0, pred_class.item()]  # Probability of predicted class

            # Store correct and incorrect prediction probabilities and indices
            if pred_class.item() == target.item():
                correct_probs.append(pred_prob.item())
                correct_indices.append(n_image)  # Store the index of the correct prediction
            else:
                incorrect_probs.append(pred_prob.item())
                incorrect_indices.append(n_image)  # Store the index of the incorrect prediction


    # Scatter plot for correct predictions
    plt.scatter(correct_indices, correct_probs, color='g', label='Correct', alpha=0.5, s=10)

    # Scatter plot for incorrect predictions
    plt.scatter(incorrect_indices, incorrect_probs, color='r', label='Incorrect', alpha=0.5, s=10)

    # Adding labels and title
    plt.xlabel('Sample Index')
    plt.ylabel('Prediction Probability')
    plt.legend(loc='upper right')
    plt.title('Scatter Plot of Prediction Probabilities for Correct and Incorrect Predictions')
    plt.show()

def evaluate_roc_curve(model, test_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.eval()  # Set model to evaluation mode

    y_true = []
    y_scores = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred_probs = torch.softmax(output, dim=1)  # Get prediction probabilities for each class

            # Assuming binary classification (e.g., attack vs benign), take the probability of the positive class
            # For multi-class classification, you would need to adjust this accordingly (e.g., select the class of interest)
            positive_class_probs = pred_probs[:, 1]  # Assuming class 1 is the "positive" class (attack)

            y_true.extend(target.cpu().numpy())  # True labels
            y_scores.extend(positive_class_probs.cpu().numpy())  # Predicted probabilities for the positive class

    # Compute ROC curve and ROC AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line (random guessing)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

    # Print the AUC score
    print(f'ROC AUC: {roc_auc:.2f}')
    print(f"Thresholds: {thresholds}")
    best_threshold = thresholds[(tpr - fpr).argmax()]
    print(f"Best threshold: {best_threshold}")

    return fpr, tpr, thresholds, roc_auc

def evaluate_model_with_threshold(model, test_loader, threshold=0.81):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.eval()  # Set model to evaluation mode

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred_probs = torch.softmax(output, dim=1)  # Get prediction probabilities for each class

            # Assuming class 1 is the positive class (attack)
            positive_class_probs = pred_probs[:, 1]  # Get the probability of the positive class

            # Apply threshold to decide class prediction
            preds = (positive_class_probs >= threshold).long()  # Convert to 0 (benign) or 1 (attack)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute the accuracy or other metrics if needed
    accuracy = (all_preds == all_labels).mean()
    print(f'Accuracy: {accuracy * 100:.2f}%')

    return all_preds, all_labels

def main():

    #steps to check before reunning this code.
    #1. save or print perturbed image .
    #2. save or print confusion matrix.
    #3. Decide epsilon, max_perturbation and perturbation_type
    #4. Select the target IDS (test_model_type) and surrogate IDS
    #5. select the data folders, label file and surrogate IDS
    #6. select GPU or CPU in load_model()
     

    #Define paths for dataset and model
    test_dataset_dir = './selected_images'
    pre_trained_model_path = './Trained_Models/custom_cnn_model_chd_resnet_ 1.pth'
    test_model_path =  './Trained_Models/custom_cnn_model_chd_resnet_ 1.pth'
    test_label_file = "selected_images.txt"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_datasets, test_loader = load_dataset(test_dataset_dir,test_label_file,device)
    print("loaded test dataset")
    # image_datasets, train_loader = load_dataset(train_dataset_dir,train_label_file,device,is_train=True)
    # print("loaded training set")
    # image_datasets,test_loader = load_dataset(dataset_dir)
    
    #laod the model
    model, test_model = load_model(pre_trained_model_path,test_model_path, test_model_type = 'res')

    # Example usage on normal test data
    evaluate_threshold_on_normal_data(model, test_loader)
    evaluate_roc_curve(model, test_loader)
    evaluate_model_with_threshold(model, test_loader, threshold=0.81)

if __name__ == "__main__":
    main()
