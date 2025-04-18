import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms, models
import numpy as np
import os
# import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score



# Define transformations and dataset paths
data_transforms = {
    'test': transforms.Compose([transforms.ToTensor()]),
    'train': transforms.Compose([transforms.ToTensor()])
}

class InceptionStem(nn.Module):
    def __init__(self):
        super(InceptionStem, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 32, stride = 1, kernel_size = 3, padding = 'same'),
            nn.Conv2d(in_channels = 32, out_channels = 32, stride = 1, kernel_size = 3, padding = 'valid'),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 1, stride = 1, padding = 'valid'),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 'same'),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 'same')
        )
    
    def forward(self, x):
        stem_out = self.stem(x)
        return stem_out
    

class InceptionResNetABlock(nn.Module):
    def __init__(self, in_channels = 128, scale=0.17):
        super(InceptionResNetABlock, self).__init__()
        self.scale = scale
        self.branch0 = nn.Conv2d(in_channels, 32, kernel_size=1, stride=1, padding='same')
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1, stride=1, padding='same'),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='same')
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1, stride=1, padding='same'),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='same'),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='same')
        )
        self.conv_up = nn.Conv2d(96, 128, kernel_size=1, stride=1, padding='same')
    
    def forward(self, x):
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        mixed = torch.cat([branch0, branch1, branch2], dim=1)
        up = self.conv_up(mixed)
        return F.relu(x + self.scale * up)
    

class ReductionA(nn.Module):
    def __init__(self, in_channels = 128):
        super(ReductionA, self).__init__()
        self.branch0 = nn.Conv2d(in_channels = in_channels, out_channels = 192, kernel_size = 3, stride = 2, padding = 'valid')
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = 96, kernel_size = 1, stride = 1, padding = 'same'),
            nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = 3, stride = 1, padding = 'same'),
            nn.Conv2d(in_channels = 96, out_channels = 128, kernel_size = 3, stride = 2, padding = 'valid')
        )
        self.branch2  = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0)

    def forward(self, x):
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        mixed = torch.cat([branch0, branch1, branch2], dim = 1)
        return mixed
    
class InceptionResNetBBlock(nn.Module):
    def __init__(self, in_channels = 448, scale = 0.10):
        super(InceptionResNetBBlock, self).__init__()
        self.scale = scale
        self.branch0 = nn.Conv2d(in_channels = in_channels, out_channels = 64, kernel_size = 1, stride = 1 , padding = 'same')
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = 64, kernel_size = 1, stride = 1, padding = 'same'),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1,3), stride = 1, padding = 'same'),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,1), stride = 1, padding = 'same')
        )
        self.conv_up = nn.Conv2d(in_channels = 128, out_channels = 448, kernel_size = 1, stride = 1, padding = 'same')


    def forward(self, x):
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        mixed = torch.cat([branch0, branch1], dim = 1)
        up = self.conv_up(mixed)
        return F.relu(x + self.scale * up)

class ReductionB(nn.Module):
    def __init__(self):
        super(ReductionB, self).__init__()
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channels = 448, out_channels = 128, kernel_size = 1, stride = 1, padding = 'same'),
            nn.Conv2d(in_channels = 128, out_channels = 192, kernel_size = 3, stride = 1, padding = 'valid')
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels = 448, out_channels = 128, kernel_size = 1, stride = 1, padding = 'same'),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 'valid')
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels = 448, out_channels = 128, kernel_size = 1, stride = 1, padding = 'same'),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 'same'),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 'valid')
        )

        self.branch3 = nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 0)

    def forward(self, x):
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        mixed = torch.cat([branch0, branch1, branch2, branch3], dim = 1)
        return mixed


# Inception-ResNet Model
class InceptionResNetV1(nn.Module):
    def __init__(self, num_classes=2):
        super(InceptionResNetV1, self).__init__()
        self.stem = InceptionStem()
        self.a_block = InceptionResNetABlock()
        self.b_block = InceptionResNetBBlock()
        self.red_a = ReductionA()
        self.red_b = ReductionB()
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(896, num_classes)
        

    def forward(self, x):
        x = self.stem(x)
        x = self.a_block(x)
        x = self.red_a(x)
        x = self.b_block(x)        
        x = self.red_b(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return F.log_softmax(x, dim = 1)


def load_labels(label_file):
    """Load image labels from the label file."""
    labels = {}
    with open(label_file, 'r') as file:
        for line in file:
            filename, label = line.strip().replace("'", "").replace('"', '').split(': ')
            labels[filename.strip()] = int(label.strip())
    return labels

def load_dataset(data_dir, label_file, is_train):
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
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=4)

    print(f'Loaded {len(images)} images.')
    return data_loader



def train_wisa(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    correct = 0
    total = 0
    running_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Track running loss
        running_loss += loss.item()
        
        # Calculate accuracy
        pred = output.argmax(dim=1, keepdim=False)  # Get the predicted class
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        #if batch_idx % 10_000 == 0:  # Adjust this to suit your dataset size
        accuracy = 100. * correct / total
        print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                  f"Loss: {loss.item():.6f} Accuracy: {accuracy:.2f}%")
    
    print("Training complete!")
    torch.save(model.state_dict(), './Trained_Models/wisa.pth')

    # Print overall training loss and accuracy for the epoch
    overall_accuracy = 100. * correct / len(train_loader.dataset)
    print(f"Epoch {epoch} Summary: Average Loss: {running_loss / len(train_loader):.6f}, "
          f"Accuracy: {overall_accuracy:.2f}%")
    
def train_model(train_loader, device,model_type):
    
    """Train the model on the training dataset."""
    if model_type == 'densenet161':
        model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, 2)
    elif model_type == 'wisa':
        model = InceptionResNetV1(num_classes=2)

   
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model = model.to(device)

    num_epochs = 10
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
    torch.save(model.state_dict(), './Trained_Models/new_d161.pth')

    return model

def test_wisa(model, device, test_loader, criterion):

    model.load_state_dict(torch.load('./Trained_Models/only_dos_d161.pth', weights_only='True'))
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=False)
            correct += pred.eq(target).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')
    
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)")
    print(f"Precision: {precision:.8f}, Recall: {recall:.8f}, F1-score: {f1:.8f}\n")
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(all_targets), yticklabels=np.unique(all_targets))
    plt.title('Confusion Matrix')
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.show()

def test_model(test_loader, device,model):

    model.load_state_dict(torch.load('./Trained_Models/new_d161.pth', weights_only='True'))
    model.eval()
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

    return all_preds, all_labels  # Return accuracy for potential further use

def evaluation_metrics(all_preds, all_labels):

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    # plt.savefig('./CF_Images_Inject20_modifygrad_d161', dpi=300)
    plt.savefig('./CF_Results/new_surr.png', dpi=300)
    plt.show()
    

    # Now you can access the true negatives and other metrics
    true_negatives = cm[0, 0]
    false_positives = cm[0, 1]
    false_negatives = cm[1, 0]
    true_positives = cm[1, 1]

    IDS_accu = accuracy_score(all_labels, all_preds) 
    IDS_prec = precision_score(all_labels, all_preds)
    IDS_recall = recall_score (all_labels,all_preds)
    IDS_F1 = f1_score(all_labels,all_preds)
    
    return IDS_accu, IDS_prec, IDS_recall, IDS_F1


def main():

    
    # Define your dataset directories
    train_dataset_dirs = 'Surrogate Dataset/train/combined_train'  # Add the paths to your other training directories
    train_label_files = 'Surrogate Dataset/train/combined_train/combined_labels.txt'
    
    test_dataset_dirs = 'Surrogate Dataset/test/combined_test' # Same for test directories
    test_label_files = 'Surrogate Dataset/test/combined_test/combined_labels.txt'
    # Define the corresponding label files for each directory
    
    

    # Set up the device (GPU or CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the test and train datasets from multiple folders
    train_loader = load_dataset(train_dataset_dirs, train_label_files,is_train=True)
    print("Loaded train dataset")

    test_loader = load_dataset(test_dataset_dirs, test_label_files,is_train=False)
    print("Loaded test dataset")
    
    model_type = 'densenet161'  # Change to the desired model type
    model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    model = model.to(device)

    # Train the model
    model = train_model(train_loader, device,model_type)

    all_preds, all_labels = test_model(test_loader, device, model )

    IDS_accu, IDS_prec, IDS_recall,IDS_F1 = evaluation_metrics(all_preds, all_labels)
    print("----------------IDS Perormance Metric----------------")
    print(f'Accuracy: {IDS_accu:.4f}')
    print(f'Precision: {IDS_prec:.4f}')
    print(f'Recall: {IDS_recall:.4f}')
    print(f'F1 Score: {IDS_F1:.4f}')
    

if __name__ == "__main__":
    main()
