import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def load_data(data_dir, batch_size=1):
    """
    Load datasets and create DataLoader for training, validation, and testing.

    Parameters:
    - data_dir: Directory containing the dataset
    - batch_size: Size of the batches for the DataLoader

    Returns:
    - dataloaders: Dictionary of DataLoader objects for train, val, and test
    - dataset_sizes: Dictionary with sizes of datasets for train, val, and test
    - class_names: List of class names
    """
    data_transforms = {
        'test': transforms.Compose([
            transforms.ToTensor(),
            # Uncomment below for normalization if needed
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=0) for x in ['test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}
    
    print("Dataset sizes:", dataset_sizes)
    
    class_names = image_datasets['test'].classes
    print("Class names:", class_names)

    return dataloaders, dataset_sizes, class_names

def load_model(model_path, num_classes):
    """
    Load a pre-trained model and modify the final layer.

    Parameters:
    - model_path: Path to the trained model weights
    - num_classes: Number of classes for the classification task

    Returns:
    - model: The modified model
    """
    model = models.resnet18(pretrained=True)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Modify the final layer
    model.load_state_dict(torch.load(model_path))  # Load the trained model weights
    model.eval()  # Set the model to evaluation mode

    return model  # Return the model on the appropriate device

def test_model(model, test_loader,device):
    """
    Test the model on the test dataset and calculate accuracy.

    Parameters:
    - model: The trained model
    - test_loader: DataLoader for the test dataset

    Returns:
    - test_accuracy: Accuracy of the model on the test dataset
    """
    all_preds = []
    all_labels = []

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to the appropriate device
        with torch.no_grad():  # Disable gradient calculation for evaluation
            outputs = model(inputs)  # Forward pass
            _, preds = torch.max(outputs, 1)  # Get predicted class labels
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    # print(f"ALL PREDS {np.unique(np.array(all_preds), return_counts = True)}")
    all_labels = np.array(all_labels)
    # print(f"ALL LABELS {np.unique(np.array(all_labels), return_counts = True)}")

    # Calculate accuracy
    test_accuracy = np.sum(all_preds == all_labels) / len(all_labels)
    print(f'Test Accuracy: {test_accuracy:.4f}')

    return test_accuracy

def main():

    data_dir = './'
    model_path = 'custom_cnn_model_new_stuffing 1.pth'
    print("Test for old_dataloader......")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load data
    dataloaders, dataset_sizes, class_names = load_data(data_dir)
    
    # print(f"CLASS NAMES: {class_names}")
    # print(f"CLASS NAMES LEN: {len(class_names)}")

    # Load model
    model = load_model(model_path, num_classes = 2)
    model = model.to(device)
    # Test the model
    
    test_model(model, dataloaders['test'],device)

if __name__ == "__main__":
    main()
