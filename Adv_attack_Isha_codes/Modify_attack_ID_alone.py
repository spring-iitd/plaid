import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.utils as vutils
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision.utils import save_image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Define transformations and dataset paths
data_transforms = {
        'test': transforms.Compose([transforms.ToTensor()]),
        'train': transforms.Compose([transforms.ToTensor()])
    }

def bit_stuff(data):
    """
    Perform bit stuffing for both `0`s and `1`s. If any bit of the same polarity exceeds five consecutive occurrences,
    it adds another bit of opposite polarity.
    """
    stuffed = ""
    count = 0
    last_bit = None  # Track the last bit to detect polarity changes

    for bit in data:
        # Increment the count if the bit matches the last one
        if bit == last_bit:
            count += 1
        else:
            count = 1  # Reset count for a new bit
            last_bit = bit  # Update the last bit tracker
        
        # Add the current bit to the stuffed data
        stuffed += bit
        
        # If we reach six consecutive bits of the same polarity, stuff the opposite bit
        if count == 5:
            stuffed += '0' if bit == '1' else '1'
            count = 0  # Reset the count after stuffing

    # print("stuffed:", stuffed)
    return stuffed

def destuff_bits(binary_string):
    """
    Removing '1' inserted after every 5 consecutive '0's in the binary string.

    Args:
        binary_string (str): Binary string to be destuffed.

    Returns:
        str: Binary string after destuffing.
    """
    result = ''
    count = 0

    i = 0
    while i < len(binary_string):
        bit = binary_string[i]
        result += bit
        if bit == '0':
            count += 1
            if count == 5:
                # Skip the next bit if it is '1'
                if i + 1 < len(binary_string) and binary_string[i + 1] == '1':
                    i += 1
                count = 0
        else:
            count = 0
        i += 1

    return result

def crc_remainder(input_bitstring, polynomial_bitstring, initial_filler):
    polynomial_bitstring = polynomial_bitstring.lstrip('0')
    len_input = len(input_bitstring)
    print("len_input",len_input)
    initial_padding = initial_filler * (len(polynomial_bitstring) - 1)
    input_padded_array = list(input_bitstring + initial_padding)
    
    while '1' in input_padded_array[:len_input]:
        cur_shift = input_padded_array.index('1')
        for i in range(len(polynomial_bitstring)):
            input_padded_array[cur_shift + i] = \
                str(int(polynomial_bitstring[i] != input_padded_array[cur_shift + i]))
                
    return ''.join(input_padded_array)[len_input:]

def denorm(batch, mean=[0.1307], std=[0.3081]):
    return batch
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

def evaluation_metrics(all_preds, all_labels,max_perturbations,perturbation_type):

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
    misclassified_attack_packets = ((all_labels == 1) & (all_preds == 0)).sum().item()

    # Total number of original attack packets (all_labels == 0)
    total_attack_packets = (all_labels == 1).sum().item()
    # total_attack_packets = 1361
    oa_asr = misclassified_attack_packets / total_attack_packets

    return tnr, mdr, oa_asr, IDS_accu, IDS_prec, IDS_recall, IDS_F1

def load_model(image_datasets, pre_trained_model_path,test_model_path, test_model_type,surr_model_type):
    # Load the pre-trained ResNet-18 model
    
    # labels = image_datasets.tensors[1]
    # unique_classes = torch.unique(labels)
    # num_classes = len(unique_classes)
    num_classes = 2
    
    if surr_model_type == 'resnet18':
        # test_model = models.resnet18(pretrained=True)
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif surr_model_type == 'densenet161':
        model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif surr_model_type == 'densenet201':
        model = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    if test_model_type == 'resnet18':
        # test_model = models.resnet18(pretrained=True)
        test_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        test_model.fc = nn.Linear(test_model.fc.in_features, num_classes)
    elif test_model_type == 'resnet50':
        # test_model = models.resnet18(pretrained=True)
        test_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        test_model.fc = nn.Linear(test_model.fc.in_features, num_classes)
    elif test_model_type == 'densenet161':
        test_model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
        test_model.classifier = nn.Linear(test_model.classifier.in_features, num_classes)
    elif test_model_type == 'densenet201':
        test_model = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
        test_model.classifier = nn.Linear(test_model.classifier.in_features, num_classes)
    else:
        test_model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        test_model.classifier[2] = nn.Linear(test_model.classifier[2].in_features, num_classes)

    #If the systen don't have GPU
    # model.load_state_dict(torch.load(pre_trained_model_path, map_location=torch.device('cpu')))
    # test_model.load_state_dict(torch.load(test_model_path, map_location=torch.device('cpu')))

    #If the system has GPU
    # model.load_state_dict(torch.load(pre_trained_model_path))
    model.load_state_dict(torch.load(pre_trained_model_path, weights_only=True))
    test_model.load_state_dict(torch.load(test_model_path, weights_only=True))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    test_model = test_model.to(device)
    
    model.eval()
    test_model.eval()

    return model, test_model

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

def load_dataset(data_dir,label_file,device,is_train=True):
    # Load datasets
    image_labels = load_labels(label_file)
    
    # Load images and create lists for images and labels
    images = []
    labels = []

    for filename, label in image_labels.items():
        img_path = os.path.join(data_dir, filename)
        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
            if is_train:
                image = data_transforms['train'](image)  # Apply training transformations
            else:
                image = data_transforms['test'](image)  # Apply testing transformations
            images.append(image)
            labels.append(label)

    # Create tensors and send them to the specified device
    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels)

    # Create DataLoader
    dataset = TensorDataset(images_tensor, labels_tensor)
    batch_size = 32 if is_train else 1  # Use larger batch size for training
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f'Loaded {len(images)} images.')
    # print(f'First image shape: {data_loader.dataset[0][0].shape}, Label: {data_loader.dataset[0][1]}')

    return dataset, data_loader

def saving_image(img, name,perturbation_type,max_perturbations):
    # to_pil = transforms.ToPILImage()
    # img = to_pil(img)
    # img.save('./Perturbed_attack_images_max_grad20/perturbed_image_{}.png'.format(name))
    save_image(img, f'./test_{perturbation_type}_densenet201_{max_perturbations}inj/perturbed_image_{name}.png')
    # img.save(name)

def print_image(img,n,pack):
    img = img.detach()
    img = img.squeeze().permute(1, 2, 0).cpu().numpy()  # Convert to numpy format
    plt.imshow(img, cmap='gray', interpolation='none')
    if n == 1:
        plt.title(f"Mask, Injection {pack})")
    elif n == 2:
        plt.title(f"Perturbed image, Injection{pack}")
    plt.show()

def calculate_crc(data):
    """
    Calculate CRC-15 checksum for the given data.
    """
    crc = 0x0000
    # CRC-15 polynomial
    poly = 0x4599

    for bit in data:
        # XOR with the current bit shifted left by 14 bits
        crc ^= (int(bit) & 0x01) << 14

        for _ in range(15):
            if crc & 0x8000:
                crc = (crc << 1) ^ poly
            else:
                crc <<= 1

        # Ensuring 15 bits
        crc &= 0x7FFF

    return crc

def print_bits_from_image(image,mask):
    # Print the bits of the perturbed image for each channel and for a specific row
    for b in range(image.shape[0]):  # Iterate over batch dimension
        # Assume you're interested in the first identified row (as an example)
        row = mask[b, 0].nonzero(as_tuple=True)[0]  
        if len(row) > 0:  # Check if any row was identified
            row = row[0].item()  # Get the first row index
            
            # Flatten the bits into a single tensor of shape (128,)
            bits = image[b, :, row, :].flatten()  # Flatten the specific row across all channels
            
            # Convert to binary representation (0s and 1s)
            binary_representation = ''.join(['1' if bit > 0.5 else '0' for bit in bits])
            print("length of binary representation:",len(binary_representation))
            print(f"Perturbed bits for batch {b}, row {row}: {binary_representation}")

def select_row_to_perturb(mask, data_grad, matched_rows,selected_rows_set):
    """
    Select the row from matched rows that has the maximum gradient in the specified mask bits.
    Avoids re-selecting rows that have already been perturbed by skipping them.

    Parameters:
        - image: The input image.
        - mask: The current mask indicating where perturbations can be applied.
        - data_grad: The gradient of the image (used to determine the regions with the highest gradient).
        - matched_rows: The list of rows that match the pattern.
        - bit_pattern: The bit pattern used to generate the mask.
        - mask_length: The length of the mask applied to each row (64 bits).
        - selected_rows_set: A set of rows that have already been selected for perturbation (to avoid re-selection).

    Returns:
        - selected_row: The row with the highest gradient to perturb.
        - updated_mask: The updated mask with only the selected row's bits active.
        - selected_rows_set: The updated set of selected rows.
    """
    gradients = []

    # # Initialize the set of selected rows if it doesn't exist
    # if selected_rows_set is None:
    #     selected_rows_set = set()

    # Loop over each matched row
    for row in matched_rows:
        # Skip rows that have already been selected
        if row in selected_rows_set:
            continue

        # Extract the gradients for the current row only in the active mask bits
        row_mask = mask[:, :, row, :].bool()  # Binary mask for this row's bits
        row_grad = data_grad[:, :, row, :]  # Gradient values for the row

        # Compute the gradient magnitude only where the mask is active
        gradient_magnitude = row_grad.abs() * row_mask
        total_gradient = gradient_magnitude.sum().item()  # Compute total gradient magnitude

        # Store the row index and total gradient magnitude
        gradients.append((row, total_gradient))

    if gradients:
        # Select the row with the maximum total gradient magnitude
        selected_row, _ = max(gradients, key=lambda x: x[1])
        # print(f"Selected row: {selected_row}, Gradient magnitude: {_}")

        # Update the mask to keep only the selected row's active bits
        updated_mask = torch.zeros_like(mask)
        updated_mask[:, :, selected_row, :] = mask[:, :, selected_row, :]

        # Add the selected row to the set of selected rows
        selected_rows_set.add(selected_row)
        # print("Updated selected rows set:", selected_rows_set)
        return selected_row, updated_mask,selected_rows_set
    else:
        # If no rows are available, return None
        updated_mask = torch.zeros_like(mask)
        return None, updated_mask,selected_rows_set

def find_max_perturbations(image,pattern_length,rgb_pattern,matched_rows):
    # If matched_rows is empty, perform the initial computation to find rows matching the pattern
    if matched_rows is None:
        # print("matched rows None")
        matched_rows = []
        
    for i in range(image.shape[2]):  # Iterate over rows in the image
        matches_pattern = torch.ones(image.shape[0], dtype=torch.bool, device=image.device)

        for j in range(pattern_length):
            r, g, b = rgb_pattern[j]
            matches_pattern &= (image[:, 0, i, j] == r) & (image[:, 1, i, j] == g) & (image[:, 2, i, j] == b)

        # Debug: Check matches for current row
        # print(f"Row {i} - Matches Pattern: {matches_pattern}")

        if matches_pattern.any():
            # Collect row indices of matching rows
            matched_rows.extend(
                [i for b in range(image.shape[0]) if matches_pattern[b]]
            )

    print("Initial matched rows:", matched_rows)
    max_perturbations = len(matched_rows)
    return matched_rows, max_perturbations

def generate_mask_modify(image, data_grad, matched_rows,selected_rows_set,bit_pattern):
    """
    Generate a mask for the image that matches the bit pattern and applies bit stuffing.
    Calls `select_row_to_perturb` to decide which row to perturb based on gradients.
    Ensures selected rows are not reused. Iterates over rows only once.
    """
    sof_len = 1
    mask_length = 11
    # print("matched_rows in generate mask modify",matched_rows)
    # Initialize matched_rows and selected_rows_set if not provided
    
    if selected_rows_set is None:
        selected_rows_set = set()

    mask = torch.zeros_like(data_grad)  # Initialize mask with zeros
    
    rgb_pattern = [(0.0, 0.0, 0.0) if bit == '0' else (1.0, 1.0, 1.0) for bit in bit_pattern]
    pattern_length = len(rgb_pattern)
    
    if not matched_rows:
        # print("No matched rows provided. Searching for rows matching the pattern.")
        matched_rows, max_perturbations = find_max_perturbations(image,pattern_length,rgb_pattern,matched_rows)

    # Filter matched_rows to exclude rows in selected_rows_set
    filtered_matched_rows = [row for row in matched_rows if row not in selected_rows_set]
    # print("Filtered matched rows:", filtered_matched_rows)

    # If no rows remain after filtering, return an empty mask
    if not filtered_matched_rows:
        # print("No rows available for perturbation. Returning empty mask.")
        return torch.zeros_like(mask), 0, matched_rows, selected_rows_set

    # Apply the mask for rows that match the pattern and are not yet selected
    for row in filtered_matched_rows:
        for b in range(image.shape[0]):
            mask[b, :, row, sof_len:sof_len + mask_length] = 1
    
    
    # print("Initial mask:")
    # print_image(mask,1,1)
    # print_bits_from_image(mask,mask)
    # Call the function to select the row with the maximum gradient from the filtered rows
    selected_row, updated_mask, selected_rows_set = select_row_to_perturb(mask, data_grad, filtered_matched_rows, selected_rows_set)

    # Mark the selected row as used
    selected_rows_set.add(selected_row)
    # print("Updated mask:")
    # print_image(updated_mask,1,1)
    # print_bits_from_image(updated_mask,updated_mask)

    return updated_mask, matched_rows, selected_rows_set

def gradient_perturbation(image, perturbed_image,mask):
    ID_len = 11
    # Ensure the identified rows' pixels are either (0, 0, 0) or (256, 256, 256)
    for b in range(image.shape[0]):

        rows = mask[b, 0].nonzero(as_tuple=True)[0]  # Identified row indices
        rows = torch.unique(rows)
        # print(rows)
        perturbation_bits = ''
        for row in rows:
            # print("image.........",image[b, :, row, 0:128])
            # print("random perturbed image.........",perturbed_image[b, :, row, 0:128])
            for col in range(1, 1 + ID_len):
                pixel_value = perturbed_image[b, :, row, col]
                dot_product_with_1 = torch.dot(pixel_value, torch.tensor([1.0, 1.0, 1.0], device=image.device))
                dot_product_with_0 = torch.dot(pixel_value, torch.tensor([0.0, 0.0, 0.0], device=image.device))
                if dot_product_with_1 >= dot_product_with_0:
                    perturbed_image[b, :, row, col] = 1.0  # Set to (256, 256, 256) in range [0, 1]
                    perturbation_bits +='1'
                else:
                    perturbed_image[b, :, row, col] = 0.0  # Set to (0, 0, 0)
                    perturbation_bits +='0'
                
            # print("projected perturbation.........",perturbed_image[b, :, row, 0:12])

            data_bits = '0' * 64
            print("length of perurbation bits",len(perturbation_bits))
            starting_bits = '0' + perturbation_bits + '0' + '0' + '0' + '1000' + data_bits
            print("length of starting bits upto data",len(starting_bits))
            
            crc_output = calculate_crc(starting_bits)
            crc_output = bin(crc_output)[2:].zfill(15)

            stuffing = starting_bits + crc_output
            # print("stuffing len",stuffing,len(stuffing),type(stuffing))
            
            stuffed_perturbation_bits = bit_stuff(stuffing)
            # print("stuffed perturatiob bits length",len(stuffed_perturbation_bits))
            
            for i,bit in enumerate(stuffed_perturbation_bits):
                value = 1.0 if bit =='1' else 0.0
                perturbed_image[b, :, row, i] = value

            # print("log3 uptill crc", perturbed_image[b, :, row, 0:len(stuffed_perturbation_bits)])
            # print("log3 crc len", perturbed_image[b, :, row, 0:len(stuffed_perturbation_bits)].shape[1])
            #ending part = (CRC del, ack, ack del, EoF, IFS)
            ending_part = '1011111111111'
            for i, bit in enumerate(ending_part):
                value = 1.0 if bit == '1' else 0.0
                perturbed_image[b, :, row, len(stuffed_perturbation_bits)+ i] = value
                
            # print("log4 ending part ", perturbed_image[b, :, row, len(stuffed_perturbation_bits):len(stuffed_perturbation_bits)+len(ending_part)])
            # print("log4 len ", perturbed_image[b, :, row, len(stuffed_perturbation_bits):len(stuffed_perturbation_bits)+len(ending_part)].shape[1])
            
            # Mark the rest of the pixels in the row as green
            for i in range(len(stuffed_perturbation_bits)+len(ending_part), 128):
                perturbed_image[b, 1, row, i] = 1.0  # Set green channel to maximum
                perturbed_image[b, 0, row, i] = 0.0
                perturbed_image[b, 2, row, i] = 0.0
            # print("log5 green portion", perturbed_image[b, :, row, len(stuffed_perturbation_bits)+len(ending_part):128])
            # print("log5 len green", perturbed_image[b, :, row,len(stuffed_perturbation_bits)+len(ending_part):128].shape[1])

            # print("Image after perturbation 1")
            # print("final image", perturbed_image[b, :, row, 0:128])
            # print_image(perturbed_image,1, 1)
            # print_bits_from_image(perturbed_image,mask)
        
        
    return perturbed_image

def fixed_id_perturbation(image, perturbed_image,mask):
    # ID_len = 11
    ID = "00100110000"
    # Ensure the identified rows' pixels are either (0, 0, 0) or (256, 256, 256)
    for b in range(image.shape[0]):

        rows = mask[b, 0].nonzero(as_tuple=True)[0]  # Identified row indices
        rows = torch.unique(rows)

        for row in rows:
            data_bits = '0' * 64
            # print("length of perurbation bits",len(perturbation_bits))
            starting_bits = '0' + ID + '0' + '0' + '0' + '1000' + data_bits
            # print("length of starting bits",len(starting_bits))
            
            crc_output = calculate_crc(starting_bits)
            crc_output = bin(crc_output)[2:].zfill(15)
            # print("crc_output length",len(crc_output))

            stuffing = starting_bits + crc_output
            # print("stuffing len",stuffing,len(stuffing),type(stuffing))
            
            stuffed_perturbation_bits = bit_stuff(stuffing)
            # print("stuffed perturatiob bits length",len(stuffed_perturbation_bits))
        
            
            for i,bit in enumerate(stuffed_perturbation_bits):
                value = 1.0 if bit =='1' else 0.0
                perturbed_image[b, :, row, i] = value

            # print("log3 uptill crc", perturbed_image[b, :, row, 0:len(stuffed_perturbation_bits)])
            # print("log3 crc len", perturbed_image[b, :, row, 0:len(stuffed_perturbation_bits)].shape[1])
            
            #ending part = (CRC del, ack, ack del, EoF, IFS)
            ending_part = '1011111111111'
            for i, bit in enumerate(ending_part):
                value = 1.0 if bit == '1' else 0.0
                perturbed_image[b, :, row, len(stuffed_perturbation_bits)+ i] = value
                
            # print("log4 ending part ", perturbed_image[b, :, row, len(stuffed_perturbation_bits):len(stuffed_perturbation_bits)+len(ending_part)])
            # print("log4 len ", perturbed_image[b, :, row, len(stuffed_perturbation_bits):len(stuffed_perturbation_bits)+len(ending_part)].shape[1])
            
            # Mark the rest of the pixels in the row as green
            for i in range(len(stuffed_perturbation_bits)+len(ending_part), 128):
                perturbed_image[b, 1, row, i] = 1.0  # Set green channel to maximum
                perturbed_image[b, 0, row, i] = 0.0
                perturbed_image[b, 2, row, i] = 0.0
            # print("log5 green portion", perturbed_image[b, :, row, len(stuffed_perturbation_bits)+len(ending_part):128])
            # print("log5 len green", perturbed_image[b, :, row,len(stuffed_perturbation_bits)+len(ending_part):128].shape[1])

            # print("Image after perturbation 1")
            # print("final image", perturbed_image[b, :, row, 0:128])
            # print_image(perturbed_image,1, 1)
            # print_bits_from_image(perturbed_image,mask)

    return perturbed_image

def fgsm_attack_modify(image,data_grad, epsilon,perturbation_type , pack,matched_rows,selected_rows_set,bit_pattern):
    # Collect the element-wise sign of the data gradient    
    sign_data_grad = data_grad.sign()
    # sign_data_grad = data_grad
    # print("matched_rows in fgsm attack modify",matched_rows)

    # Create a mask to apply sign data grad only in the rows with max gradient magnitude
    mask,matched_rows,selected_rows_set = generate_mask_modify(image, data_grad,matched_rows,selected_rows_set,bit_pattern)
    sign_data_grad = sign_data_grad * mask
    
    # print("Image before perturbation")
    # print_image(image,1,pack)
    # print_bits_from_image(image,mask)
    # print_image(mask,1,pack)
    # print("before perturbation image.........",image[b, :, row, 0:128])
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad

    if perturbation_type == "Gradient":
        perturbed_image = gradient_perturbation(image, perturbed_image,mask)
    elif perturbation_type == "Targetted":
        perturbed_image = fixed_id_perturbation(image, perturbed_image,mask)
    
    # Return the perturbed image
     # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image,matched_rows,selected_rows_set

def apply_fgsm_and_check( pack, test_model,target,data_grad,data_denorm,ep,perturbation_type,matched_rows,selected_rows_set,bit_pattern):
    
    # print("matched_rows in apply_fgsm and check",matched_rows)
    # perturbed_data = fgsm_attack_valid(data_denorm, data_grad,ep,perturbation_type,pack)
    perturbed_data,matched_rows,selected_rows_set = fgsm_attack_modify(data_denorm,data_grad, ep,perturbation_type , pack,matched_rows,selected_rows_set,bit_pattern)
    # print_image(perturbed_data,2,pack)
    
    if perturbed_data is None:
        print("No more space to inject")
        return False, pack, final_pred, perturbed_data 
    
    with torch.no_grad():
        output = test_model(perturbed_data)


    pred_probs = torch.softmax(output, dim=1)
    print("Probability of prediction",pred_probs)
    # Get the predicted class index
    final_pred = output.max(1, keepdim=True)[1] # index of the maximum log-probability
    # print("predicted, label ",final_pred.item(), target.item())

    
    #for 0-benign, 1-attack
    if final_pred.item() == target.item():
        # print("Perturbation {} not successful. Injecting more perturbation.".format(pack))
        return True, pack+1, final_pred, perturbed_data,matched_rows,selected_rows_set  # Indicate that we need to reapply
    else:
        # print("Perturbation {} successful. No more injection needed, return pack as final perturbation".format(pack))
        return False, pack, final_pred, perturbed_data,matched_rows,selected_rows_set  # Indicate that we can stop
    
def Attack_procedure(model, test_model, device, test_loader, perturbation_type, ep, max_perturbations):
    pack = 1
    all_preds = []
    all_labels = []
    n_image = 1
    # target_model = test_model
    bit_pattern = "0000010000010000011000"
    rgb_pattern = [(0.0, 0.0, 0.0) if bit == '0' else (1.0, 1.0, 1.0) for bit in bit_pattern]
    pattern_length = len(rgb_pattern)
    for data, target in test_loader:
        # print(f"Current target shape: {target.shape}, value: {target}")
        data, target = data.to(device), target.to(device)
        
        # If target is a 1D tensor, no need for item()
        current_target = target[0] if target.dim() > 0 else target

        # Initialize predictions for benign images (target=0)
        initial_output = model(data)
        final_pred = initial_output.max(1, keepdim=True)[1]
        perturbation_count = 1
        # Only perform perturbation for attack images (target=1)
        if current_target == 1:
            print("Image no:", n_image, "(Attack image)")
            pack = 1
            
            data.requires_grad = True
            model.eval()
            
            initial_output = model(data)
            loss = F.nll_loss(initial_output, target)
            
            model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            
            data_denorm = denorm(data)
            continue_perturbation = True
            matched_rows = None
            selected_rows_set = None

            _,max_perturbations = find_max_perturbations(data_denorm,pattern_length,rgb_pattern,matched_rows)
            print("max_perturbations",max_perturbations)

            while continue_perturbation and perturbation_count <= max_perturbations:
                
                if continue_perturbation:
                    perturbed_data = data_denorm.clone().detach().to(device)
                    perturbed_data.requires_grad = True
                    model.eval()
                    perturbed_output = model(perturbed_data)
                    new_loss = F.nll_loss(perturbed_output, target)
                    
                    model.zero_grad()
                    new_loss.backward()
                    data_grad = perturbed_data.grad.data
                
                # print("matched_rows in attack procedure",matched_rows)
                continue_perturbation, pack, final_pred, data_denorm,matched_rows,selected_rows_set = apply_fgsm_and_check(pack, test_model, target, data_grad, perturbed_data, ep, perturbation_type,matched_rows,selected_rows_set,bit_pattern)
                perturbation_count += 1
                # print("perturbation count in attack procedure",perturbation_count)
                # print("max perturbations in attack procedure.....",max_perturbations)
                # print("continue_perturbation in attack procedure",continue_perturbation)      
                # print("matched rows in attack procedure.....",matched_rows)  
                # print("selected rows in attack procedure.....",selected_rows_set)



            # saving_image(data_denorm,n_image,perturbation_type,max_perturbations)
        else:
            data.requires_grad = True
            model.eval()
            initial_output = model(data)
            final_pred = initial_output.max(1, keepdim=True)[1]
            print("Image no:", n_image, "(Benign image - skipping perturbation)")
            # saving_image(data,n_image,perturbation_type,max_perturbations)
        
        # target_output = target_model(data_denorm)
        # pp = torch.softmax(target_output, dim=1)
        # print(pp)
        # fp = target_output.max(1, keepdim=True)[1]
        # print("target model final preds",fp)
        print("final no of perturbations:", perturbation_count-1)
        print("Image {}, truth_labels {}, final_pred {} ".format(n_image,target.item(), final_pred.cpu().numpy()))
        n_image += 1
        all_preds.extend(final_pred.cpu().numpy())
        all_labels.extend(target.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    # with open("perturb attack_preds.txt", "w") as file: file.write("\n".join(all_preds.squeeze()) + "\n")
    return all_preds.squeeze(), all_labels
    

def main():

    #steps to check before reunning this code.
    #1. save or print perturbed image .
    #2. save or print confusion matrix.
    #3. Decide epsilon, max_perturbation and perturbation_type
    #4. Select the target IDS (test_model_type) and surrogate IDS
    #5. select the data folders, label file and surrogate IDS
    #6. select GPU or CPU in load_model()

    # Check if the correct number of arguments is provided
    if len(sys.argv) != 2:
        print("Usage: python file_name.py <PerturbationType>")
        sys.exit(1)

    # Read the perturbation type from the command-line argument
    perturbation_type = sys.argv[1]


    test_model_type = 'densenet161'
    surr_model_type='densenet161'

    #Define paths for dataset and model
    test_dataset_dir = './selected_images'
    # test_dataset_dir = './image'
    # train_dataset_dir = './selected_images'
    # surr_model_path = './Trained_Models/custom_cnn_model_chd_resnet_ 1.pth'
    surr_model_path = "./Trained_Models/custom_cnn_model_chd_densenet_161.pth"
    # feedback_model_path =  './Trained_Models/custom_cnn_model_chd_resnet_ 1.pth'
    test_model_path = "./Trained_Models/custom_cnn_model_chd_densenet_161.pth"
    # test_model_path = "./Trained_Models/custom_cnn_model_chd_densenet_201.pth"
    # test_label_file = "./image/image.txt"
    test_label_file = "selected_images.txt"
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_datasets, test_loader = load_dataset(test_dataset_dir,test_label_file,device,is_train=False)
    print("loaded test dataset")
    # image_datasets, train_loader = load_dataset(train_dataset_dir,train_label_file,device,is_train=True)
    # print("loaded training set")
    # image_datasets,test_loader = load_dataset(dataset_dir)
    
    #laod the model
    model, test_model = load_model(image_datasets, surr_model_path,test_model_path, test_model_type ,surr_model_type)

    
    # Define the parameters
    epsilon = 1
    # perturbation_type = "Gradient"   
   # List of max_perturbations to iterate over
    max_perturbations_list = [0]
    # max_perturbations_list = [1, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60]

    # Loop through the list of max_perturbations
    for max_perturbations in max_perturbations_list:
        print("--------------------------------")
        print(f"Testing with max_perturbations  {max_perturbations} and perturbation_type {perturbation_type}")

        # Call the attack procedure 
        preds, labels = Attack_procedure(model, test_model, device, test_loader, perturbation_type, epsilon, max_perturbations)
        
        print("Labels:", labels)
        print("Predictions:", preds)
        
        
        tnr, mdr, oa_asr, IDS_accu, IDS_prec, IDS_recall,IDS_F1 = evaluation_metrics(preds, labels,max_perturbations,perturbation_type)
        print(f'Accuracy: {IDS_accu:.4f}')
        print(f'Precision: {IDS_prec:.4f}')
        print(f'Recall: {IDS_recall:.4f}')
        print(f'F1 Score: {IDS_F1:.4f}')
        print("TNR:", tnr)
        print("MDR:", mdr)
        print("OA_ASR:", oa_asr)



if __name__ == "__main__":
    main()
