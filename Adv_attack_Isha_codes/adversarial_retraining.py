import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms, models
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

data_transforms = {
    'test': transforms.Compose([transforms.ToTensor()]),
    'train': transforms.Compose([transforms.ToTensor()])
}

def load_perturbed(attack_perturbed_path, benign_perturbed_path, total_count, is_train = False):
    
    images = []
    labels = []
    
    
    paths = [attack_perturbed_path]

    for path in paths:

        print(path)

        if path == attack_perturbed_path:
            label = 1
        else:
            label = 0

        img_count = 0

        for img in os.listdir(path):
            
            if img_count <= total_count:
                # print(img_count)
            
                img_path = os.path.join(path, img)
                img_count += 1
                if os.path.exists(img_path):
                    image = Image.open(img_path).convert("RGB")
                    image = data_transforms["train"](image)
                    # print(image.shape)
                    images.append(image)
                    labels.append(label)
            else:
                break

    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels)
    dataset = TensorDataset(images_tensor, labels_tensor)
    batch_size = 32 if is_train else 1
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train)

    return data_loader


def pretrained_eval(test_loader, model_name, device):

    num_classes = 2
     
    if model_name == 'resnet':
         model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
         model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'densenet':
        model = models.densenet161(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        model = models.convnext_base(pretrained=True)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
   
    model_path = f'run_on_server/custom_cnn_model_chd_{model_name}_.pth'
    
    model.load_state_dict(torch.load(model_path))

    model.eval()
    model = model.to(device)

    # Initialize lists to store predictions and labels
    all_preds = []
    all_labels = []

    # Evaluate the model on the test dataset
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to the appropriate device
        with torch.no_grad():  # Disable gradient calculation for evaluation
            outputs = model(inputs)  # Forward pass
            _, preds = torch.max(outputs, 1)  # Get predicted class labels
        
        # Store predictions and labels
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate accuracy
    test_accuracy = np.sum(all_preds == all_labels) / len(all_labels)
    print(f'Test Accuracy: {test_accuracy:.4f}')

    return test_accuracy  # Return accuracy for potential further use


def load_labels(label_file):
    """Load image labels from the label file."""
    labels = {}
    with open(label_file, 'r') as file:
        for line in file:
            filename, label = line.strip().replace("'", "").replace('"', '').split(': ')
            labels[filename.strip()] = int(label.strip())
    return labels

def load_dataset(data_dir, label_file, device, num_workers, is_train=True):
    """Load datasets and create DataLoader."""
    image_labels = load_labels(label_file)
    images = []
    labels = []

    for filename, label in image_labels.items():
        img_path = os.path.join(data_dir, filename)
        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
            image = data_transforms['train' if is_train else 'test'](image)
            images.append(image)
            labels.append(label)

    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels)
    dataset = TensorDataset(images_tensor, labels_tensor)
    batch_size = 32 if is_train else 1
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers)

    print(f'Loaded {len(images)} images.')
    return data_loader

def load_and_finetune_model(train_loader, device, num_images, model_name='resnet'):
    """Load a pretrained model, fine-tune the final layers, and train it further."""

    checkpoint_path = f'run_on_server/custom_cnn_model_chd_{model_name}_.pth'
    
    num_classes = 2

    if model_name == 'resnet':
        # model = models.resnet18(xweights=models.ResNet18_Weights.DEFAULT)
        # num_features = model.fc.in_features
        # model.fc = nn.Linear(num_features, num_classes)

        model = models.resnet50(pretrained = False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        
    elif model_name == 'convnext':
        model = models.convnext_base(pretrained=True)
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, num_classes)
        
    elif model_name == 'densenet':
        model = models.densenet161(pretrained=True)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)
        
    else:
        raise ValueError("Model type not included in code")
    
    # Load the previously saved weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # Freeze all layers except for the final fully connected layer
    for param in model.parameters():
        param.requires_grad = False
        
    if model_name == 'resnet':
        for param in model.fc.parameters():
            param.requires_grad = True
            
    elif model_name == 'convnext':
        for param in model.classifier[-1].parameters():
            param.requires_grad = True
            
    elif model_name == 'densenet':
        for param in model.classifier.parameters():
            param.requires_grad = True
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0007, momentum=0.65)

    model = model.to(device)

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    print("Fine-tuning complete!")
    torch.save(model.state_dict(), f'../scratch/finetuned_model_{num_images}_{model_name}.pth')

    return model

# def test_model(test_loader, device, model):
#     """
#     Load a pre-trained model, evaluate it on the test dataset, and calculate accuracy.

#     Parameters:
#     - test_loader: DataLoader for the test dataset
#     - model_path: Path to the trained model file

#     Returns:
#     - test_accuracy: Accuracy of the model on the test dataset
#     """

#     model.eval()
#     model = model.to(device)

#     # Initialize lists to store predictions and labels
#     all_preds = []
#     all_labels = []

#     # Evaluate the model on the test dataset
#     for inputs, labels in test_loader:
#         inputs, labels = inputs.to(device), labels.to(device)  # Move data to the appropriate device
#         with torch.no_grad():  # Disable gradient calculation for evaluation
#             outputs = model(inputs)  # Forward pass
#             _, preds = torch.max(outputs, 1)  # Get predicted class labels
        
#         # Store predictions and labels
#         all_preds.extend(preds.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())

#     # Convert lists to numpy arrays
#     all_preds = np.array(all_preds)
#     all_labels = np.array(all_labels)

#     # Calculate accuracy
#     test_accuracy = np.sum(all_preds == all_labels) / len(all_labels)
#     print(f'Test Accuracy: {test_accuracy:.4f}')

#     return test_accuracy  # Return accuracy for potential further use


def test_model(test_loader, device, model, model_type, num_samples, data_type, save_roc_path="roc_curves_retraining_attack", save_roc = True):
    """
    Load a pre-trained model, evaluate it on the test dataset, calculate accuracy, and save the AUC/ROC curve.

    Parameters:
    - test_loader: DataLoader for the test dataset
    - device: Device to use for evaluation (e.g., "cuda" or "cpu")
    - model: Trained model to evaluate
    - save_roc_path: File path to save the ROC curve image

    Returns:
    - test_accuracy: Accuracy of the model on the test dataset
    - auc_score: AUC score of the model on the test dataset
    """
    model.eval()
    model = model.to(device)

    # Initialize lists to store predictions and labels
    all_probs = []  # Store probabilities for ROC curve
    all_preds = []
    all_labels = []

    # Evaluate the model on the test dataset
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to the appropriate device
        with torch.no_grad():  # Disable gradient calculation for evaluation
            outputs = model(inputs)  # Forward pass
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Get probabilities for the positive class
            _, preds = torch.max(outputs, 1)  # Get predicted class labels
        
        # Store predictions, probabilities, and labels
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Convert lists to numpy arrays
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate accuracy
    test_accuracy = np.sum(all_preds == all_labels) / len(all_labels)
    print(f'Test Accuracy: {test_accuracy:.4f}')

    # Calculate AUC and ROC curve
    if save_roc:
        auc_score = roc_auc_score(all_labels, all_probs)
        fpr, tpr, _ = roc_curve(all_labels, all_probs)

        print(f'AUC Score: {auc_score:.4f}')

        # Plot and save ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid()
        plt.savefig(os.path.join(save_roc_path, f'{model_type}_{num_samples}_{data_type}_retrain_curve.png'))
        print(f'ROC curve saved to {save_roc_path}')
        plt.close()
    else:
        pass



def main():

    # retrain_samples = [500, 1000, 1500, 2000, 3000]

    retrain_samples = [100, 250, 500, 750, 1000, 1361] 
    
    attack_perturbed_path = "../scratch/Perturbed_attack_images_max_grad20_selected"
    benign_perturbed_path = '../scratch/perturbed_benign_images/20'
    
    test_dataset_dir = './selected_images'
    test_label_file = "selected_images.txt"  # Change to correct label file if needed

    model_type = 'densenet'
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.version.cuda)  # Check the CUDA version
    print(torch.cuda.get_device_name(0))  # Get the name of the GPU
    # Load train and test datasets

## Before training
##      i)  Evaluate the model on only perturbed images
##      ii) Evaluate the model on 'selected_images'
 
 
## 4) Train on the loaded images and files
 
## 5) After training
##      i)  Evaluate the model on only perturbed images
##      ii) Evaluate the model on 'selected_images'


    perturbed_data_test = load_perturbed(attack_perturbed_path, benign_perturbed_path, 5000, is_train = False)
    test_loader = load_dataset(test_dataset_dir, test_label_file, device,num_workers=0, is_train=False)

    print("PRETRAINED EVAL RESULT ON PERTURBED DATA")
    pretrained_eval(perturbed_data_test, model_type, device)

    print("-----------x-----------")

    print("PRETRAINED EVAL RESULT ON NORMAL DATA")
    pretrained_eval(test_loader, model_type, device)

    for num_samples in retrain_samples:

        train_loader = load_perturbed(attack_perturbed_path, benign_perturbed_path, num_samples, is_train = True)
        
        print(f'TRAINING ON {num_samples} PERTURBED SAMPLES')

        finetuned_model = load_and_finetune_model(train_loader, device, num_samples, model_name=model_type)


        print("FINETUNED EVAL RESULT ON PERTURBED DATA")
        test_model(perturbed_data_test, device, finetuned_model, model_type, num_samples, 'perturbed', save_roc = False)

        print("-----------x-----------")

        print("FINETUNED EVAL RESULT ON NORMAL DATA")
        test_model(test_loader, device, finetuned_model, model_type, num_samples, 'normal')




    # # Train the model
    # model = train_model(train_loader, device, model_type)
    # test_model(test_loader, device, model)

if __name__ == "__main__":
    main()


## TODO:
### 1) Load images and files.
## attack perturbed image_path : scratch/Perturbed_attack_images_max_grad20
## attack perturbed labels: [all 1s] 
## benign perturbed image_path: scratch/perturbed_benign_images/20
## benign perturbed labels : [all 0s]


## 3) Before training
##      i)  Evaluate the model on only perturbed images
##      ii) Evaluate the model on 'selected_images'


## 4) Train on the loaded images and files

## 5) After training
##      i)  Evaluate the model on only perturbed images
##      ii) Evaluate the model on 'selected_images'

## Do it for resnet, convnext, densenet

