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

def train_model(train_loader, device, model_name = 'resnet'):
    """Train the model on the training dataset."""
    # model = models.resnet18(weights='IMAGENET1K_V1')  # Use the updated way to load weights
    
    if model_name == 'resnet':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        for name, param in model.named_parameters():
            if "fc" in name:  
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        num_classes = 2
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == 'convnext':
        model = models.convnext_base(pretrained = True)

        for param in model.parameters():
            param.requires_grad = False

        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, 2) 
    
    elif model_name == 'densenet':
        model = models.densenet169(pretrained=True)
        # model = models.densenet121(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, 2) 
    
    else:
        raise ValueError("Model type not included in code")
    

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model = model.to(device)

    num_epochs = 50
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

    print("Training complete!")
    # torch.save(model.state_dict(), f'custom_cnn_model_chd_{model_name}_new_Dataset.pth')

    return model


def test_model(test_loader, device, model, model_type, save_roc_path="roc_curves"):
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
    plt.savefig(os.path.join(save_roc_path, f'{model_type}_curve.png'))
    print(f'ROC curve saved to {save_roc_path}')
    plt.close()





def main():
    train_dataset_dir = '../../scratch/new_imgs/DoS_images/'
    test_dataset_dir = './selected_images'

    train_label_file = "../../scratch/new_imgs/DoS_labels.txt"
    test_label_file = "selected_images.txt"  # Change to correct label file if needed

    model_type = 'resnet'
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.version.cuda)  # Check the CUDA version
    print(torch.cuda.get_device_name(0))  # Get the name of the GPU
    # Load train and test datasets
    test_loader = load_dataset(test_dataset_dir, test_label_file, device,num_workers=0, is_train=False)
    print("Loaded test dataset")
    train_loader = load_dataset(train_dataset_dir, train_label_file, device,num_workers=4, is_train=True)
    print("Loaded training set")

    # Train the model
    model = train_model(train_loader, device, model_type)
    test_model(test_loader, device, model, model_type)

if __name__ == "__main__":
    main()
