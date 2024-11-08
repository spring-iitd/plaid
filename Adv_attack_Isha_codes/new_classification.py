import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, TensorDataset

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

    data_transforms = {
        'test': transforms.Compose([
            transforms.ToTensor(),
            # Uncomment below for normalization if needed
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
     
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
        print(inputs,labels)
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to the appropriate device
        with torch.no_grad():  # Disable gradient calculation for evaluation
            outputs = model(inputs)  # Forward pass
            _, preds = torch.max(outputs, 1)  # Get predicted class labels
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    print(f"ALL PREDS {np.unique(np.array(all_preds), return_counts = True)}")
    all_labels = np.array(all_labels)
    print(f"ALL LABELS {np.unique(np.array(all_labels), return_counts = True)}")

    # Calculate accuracy
    test_accuracy = np.sum(all_preds == all_labels) / len(all_labels)
    print(f'Test Accuracy: {test_accuracy:.4f}')

    return test_accuracy

def main():

    data_dir = './test_selected_images'
    label_file = 'test_selected_images.txt'
    model_path = 'custom_cnn_model_new_stuffing 1.pth'
    # model_path = '../run_on_server/custom_cnn_model_chd_resnet_.pth'
    print("Test for new_dataloader......")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load data
    image_datasets, test_loader = load_dataset(data_dir,label_file,device)

    # dataloaders, dataset_sizes, class_names = load_data(data_dir)
    
    # Load model
    model = load_model(model_path, num_classes=2)
    print("Loaded model")
    model = model.to(device)
    # Test the model
    
    test_model(model, test_loader,device)

if __name__ == "__main__":
    main()
