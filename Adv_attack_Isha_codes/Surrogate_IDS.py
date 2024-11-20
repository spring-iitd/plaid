import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms, models
import numpy as np
import os
from PIL import Image

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

def train_model(train_loader, device):
    """Train the model on the training dataset."""
    model = models.resnet18(weights='IMAGENET1K_V1')  # Use the updated way to load weights
    num_classes = 2
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model = model.to(device)

    num_epochs = 2
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
    torch.save(model.state_dict(), 'custom_cnn_model_chd_resnet_.pth')

    return model

def test_model(test_loader, device, model):
    """
    Load a pre-trained model, evaluate it on the test dataset, and calculate accuracy.

    Parameters:
    - test_loader: DataLoader for the test dataset
    - model_path: Path to the trained model file

    Returns:
    - test_accuracy: Accuracy of the model on the test dataset
    """

    # # Load the pre-trained ResNet18 model
    # model = models.resnet18(pretrained=True)
    # num_classes = 2  # Update based on your specific classification task
    # model.fc = nn.Linear(model.fc.in_features, num_classes)  # Modify the final layer
    # model.load_state_dict(torch.load(model_path))  # Load the trained model weights
    # model.eval()  # Set the model to evaluation mode

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model.to(device)  # Move model to the appropriate device

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



def main():
    train_dataset_dir = './CHD_images'
    test_dataset_dir = './selected_images'
    train_label_file = "CHD_images.txt"
    test_label_file = "selected_images.txt"  # Change to correct label file if needed
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.version.cuda)  # Check the CUDA version
    print(torch.cuda.get_device_name(0))  # Get the name of the GPU
    # Load train and test datasets
    test_loader = load_dataset(test_dataset_dir, test_label_file, device,num_workers=0, is_train=False)
    print("Loaded test dataset")
    train_loader = load_dataset(train_dataset_dir, train_label_file, device,num_workers=4, is_train=True)
    print("Loaded training set")

    # Train the model
    model = train_model(train_loader, device)
    test_model(test_loader, device, model )

if __name__ == "__main__":
    main()
