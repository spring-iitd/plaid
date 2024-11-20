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
from torchvision.utils import save_image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Define transformations and dataset paths
data_transforms = {
        'test': transforms.Compose([transforms.ToTensor()]),
        'train': transforms.Compose([transforms.ToTensor()])
    }


def bit_stuff(data):
    stuffed = ""
    count = 0
    for bit in data:
        if bit == '1':
            count += 1
            stuffed += bit
            if count == 5:
                stuffed += '0'
                count = 0
        else:
            stuffed += bit
            count = 0
    return stuffed

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

def load_model(image_datasets, pre_trained_model_path,test_model_path, test_model_type,surr_model_type='dense'):
    # Load the pre-trained ResNet-18 model
    
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

def saving_image(img, name):
    # to_pil = transforms.ToPILImage()
    # img = to_pil(img)
    # img.save('./Perturbed_attack_images_max_grad20/perturbed_image_{}.png'.format(name))
    save_image(img, f'./test_random_resnet_feedback_1inj/perturbed_image_{name}.png')
    # img.save(name)

# def save_image(perturbed_data, n_image):
#     perturbed_data = perturbed_data.detach()
#     perturbed_image_np = perturbed_data.squeeze().permute(1, 2, 0).cpu().numpy()  # Convert to numpy format
    
#     # Convert numpy array to an image
#     image = Image.fromarray((perturbed_image_np * 255).astype(np.uint8))  # Scale to [0, 255] if necessary
#     image.save('./Perturbed_attack_images_max_grad20/perturbed_image_{}.png'.format(n_image))

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

def compute_row_gradient_magnitude(data_grad, row_idx):

    """Computes the gradient magnitude for a specific row in the data gradient."""
    return data_grad[:, :, row_idx, :].abs().sum(dim=(1, 2))

def update_max_grad(row_grad_magnitude, max_grad, max_grad_row, row_idx, all_green):

    """Updates the row with maximum gradient magnitude if all pixels in the row are green."""
    update_mask = (row_grad_magnitude > max_grad) & all_green
    max_grad = torch.where(update_mask, row_grad_magnitude, max_grad)
    max_grad_row = torch.where(update_mask, torch.tensor(row_idx, device=max_grad.device), max_grad_row)
    return max_grad, max_grad_row

def create_mask_for_max_grad_row(mask, max_grad_row, image_shape):
    """Creates a mask that applies only to the identified rows with maximum gradient."""
    for b in range(image_shape[0]):
        mask[b, :, max_grad_row[b], :] = 1  # Applying on all columns of the identified row
    return mask

def initialize_max_grad_variables(batch_size, num_rows, device):
    """Initializes tensors for tracking the maximum gradient and corresponding row index."""
    max_grad = torch.zeros(batch_size, device=device)
    max_grad_row = torch.zeros(batch_size, dtype=torch.long, device=device)
    return max_grad, max_grad_row

def extract_color_channels(image):
    """Extracts the red, green, and blue channels from an image tensor."""
    red_channel = image[:, 0, :, :]
    green_channel = image[:, 1, :, :]
    blue_channel = image[:, 2, :, :]
    return red_channel, green_channel, blue_channel

def create_green_mask(red_channel, green_channel, blue_channel):
    """Creates a mask for rows where all pixels are exactly (0, 1, 0), i.e., green."""
    return (red_channel == 0) & (green_channel == 1) & (blue_channel == 0)

def find_rows_with_green(green_mask):
    """Finds rows that contain green pixels by summing along the width dimension."""
    No_green_row = False
    row_sums = green_mask.sum(dim=-1)
    green_rows = (row_sums > 0).nonzero(as_tuple=True)[1]

    if green_rows.numel() == 0:  # If no green rows found
        No_green_row = True
        
    return green_rows, No_green_row

def select_random_rows(rows_with_green, numberofrows):
    """Randomly selects a specified number of rows from the rows that contain green pixels."""
    if len(rows_with_green) > numberofrows:
        selected_rows = torch.randperm(len(rows_with_green))[:numberofrows]
        return rows_with_green[selected_rows]
    else:
        return rows_with_green

def initialize_mask(image):
    """Initializes a mask of zeros with the same dimensions as the input image."""
    mask = torch.zeros_like(image, dtype=torch.float)
    # print("Printing mask-----------",torch.all(mask == 0))
    return mask

def create_mask(mask, selected_rows):
    """Sets the selected rows in the mask to 1."""
    for row in selected_rows:
        mask[:, :, row, :] = 1.0
    return mask

def generate_multiple_mask_random(image, pack):
    red_channel, green_channel, blue_channel = extract_color_channels(image)
    green_mask = create_green_mask(red_channel, green_channel, blue_channel)
    rows_with_green, No_green_row = find_rows_with_green(green_mask)
    if No_green_row:
        return None
    selected_rows = select_random_rows(rows_with_green, pack)
    mask = initialize_mask(image)
    mask = create_mask(mask, selected_rows)
    
    return mask

def generate_max_grad_mask(image, data_grad):
    # Assuming 'image' is of shape [batch_size, 3, 128, 128]
    # We need to identify the green channel which is the 2nd channel in this format
    
    red_channel, green_channel, blue_channel = extract_color_channels(image)
    
    green_mask = create_green_mask(red_channel, green_channel, blue_channel)

    max_grad, max_grad_row = initialize_max_grad_variables(green_channel.shape[0], green_channel.shape[1], image.device)

    for i in range(green_channel.shape[1]):  # iterate over rows
        # Check if all pixels in the row are green
        all_green = green_mask[:, i, :].all(dim=1)

        # Compute gradient magnitude for the row
        row_grad_magnitude = compute_row_gradient_magnitude(data_grad, i)

        max_grad, max_grad_row = update_max_grad(row_grad_magnitude, max_grad, max_grad_row, i, all_green)

    # Create a mask to apply the sign data gradient only in the identified rows with max gradient
    mask = initialize_mask(data_grad)
    mask = create_mask_for_max_grad_row(mask, max_grad_row, image.shape)
    
    return mask

def apply_constraint(image, mask,perturbed_image ):
     # Ensure the identified row's pixels are modified according to the fixed pattern and CRC bit stuffing
    for b in range(image.shape[0]):
        row = mask[b, 0].nonzero(as_tuple=True)[0]  # Identified row index
        if len(row) > 0:  # Only apply if a row is identified
            row = row[0].item()
            # print("before perturbation",perturbed_image[b, :, row, :])
            # fixed_pattern = "0 000010 00100110000 01 0 0 1000"
            #fixed_pattern = (sof, id, RTR, IDE bit, r0, DLC)
            fixed_pattern = "00010011000001001000"
            for i, bit in enumerate(fixed_pattern):
                value = 1.0 if bit == '1' else 0.0
                perturbed_image[b, :, row, i] = value
                # colored black or white
                # Get the edited part and its length
            
            # edited_part = perturbed_image[b, :, row, :len(fixed_pattern)]
            # length_of_edited_part = edited_part.shape[1]  # Length in terms of pixels

            # Print the edited part and its length
            # print("Edited part:", edited_part)
            # print("Length of edited part:", length_of_edited_part)

          
            # perturbation_bits = ''.join(['1' if perturbed_image[b, :, row, len(fixed_pattern) + i] > 0.5 else '0' for i in range(64)])
            perturbation_bits = ''
            # print("Perturbation bits:", perturbation_bits,len(perturbation_bits))
            
            for col in range(len(fixed_pattern), len(fixed_pattern)+64):
                pixel_value = perturbed_image[b, :, row, col]
                dot_product_with_1 = torch.dot(pixel_value, torch.tensor([1.0, 1.0, 1.0], device=image.device))
                dot_product_with_0 = torch.dot(pixel_value, torch.tensor([0.0, 0.0, 0.0], device=image.device))
                if dot_product_with_1 >= dot_product_with_0:
                    perturbed_image[b, :, row, col] = 1.0  # Set to (256, 256, 256) in range [0, 1]
                    perturbation_bits +='1'
                else:
                    perturbed_image[b, :, row, col] = 0.0  # Set to (0, 0, 0)
                    perturbation_bits +='0'
             
            # print("log2 data", perturbed_image[b, :, row, len(fixed_pattern):84])
            # print("log2 data length", perturbed_image[b, :, row, len(fixed_pattern):84].shape[1])
            # print("perturbation_bits",perturbation_bits)
            # Calculate CRC (sof, id,rtr, idebit, ro, dlc,data ) crc is calculated on raw data not he bit stuffed data
            crc_input = '0' + '00100110000' + '0' + '0' + '0' + '1000' + perturbation_bits
            crc_output = calculate_crc(crc_input)
            crc_output = bin(crc_output)[2:].zfill(15)
            # crc_output = crc_remainder(crc_input, '100000111', '0')
            bit_stuffed_crc = bit_stuff(crc_output[:15])
            
            # Apply bit-stuffed CRC to the next 15 pixels
            for i, bit in enumerate(bit_stuffed_crc):
                value = 1.0 if bit == '1' else 0.0
                perturbed_image[b, :, row, len(fixed_pattern) + len(perturbation_bits) + i] = value

            # print("log3 crc", perturbed_image[b, :, row, 84:84+15])
            # print("log3 crc len", perturbed_image[b, :, row, 84:84+len(bit_stuffed_crc)].shape[1])
            
            #ending part = (CRC del, ack, ack del, EoF, IFS)
            ending_part = '1011111111111'
            for i, bit in enumerate(ending_part):
                value = 1.0 if bit == '1' else 0.0
                perturbed_image[b, :, row, len(fixed_pattern) + len(perturbation_bits)+ len(bit_stuffed_crc)+ i] = value
                # perturbed_image[b, 1, row, len(fixed_pattern) + len(perturbation_bits) +len(ending_part)+ i] = value
                # perturbed_image[b, 2, row, len(fixed_pattern) + len(perturbation_bits) +len(ending_part)+ i] = value
                # print(perturbed_image[b, :, row, len(fixed_pattern) + 64 +len(ending_part)+ i])
            
            # print("log4 ending part ", perturbed_image[b, :, row, 84+15:84+15+len(ending_part)])
            # print("log4 len ", perturbed_image[b, :, row, 84+len(bit_stuffed_crc):84+len(bit_stuffed_crc)+len(ending_part)].shape[1])
            # Mark the rest of the pixels in the row as green
            for i in range(len(fixed_pattern) + len(perturbation_bits) +len(bit_stuffed_crc)+len(ending_part), 128):
                perturbed_image[b, 1, row, i] = 1.0  # Set green channel to maximum
                perturbed_image[b, 0, row, i] = 0.0
                perturbed_image[b, 2, row, i] = 0.0
            
            # print("log5 green portion", perturbed_image[b, :, row, 84+len(bit_stuffed_crc)+len(ending_part):128])
            # print("log5 len green", perturbed_image[b, :, row,84+len(bit_stuffed_crc)+ len(ending_part):128].shape[1])

            # print("packet length",len(fixed_pattern) + 64 + len(ending_part)+len(bit_stuffed_crc))
            # print(len(fixed_pattern))
            # print("64")
            # print(len(ending_part))
            # print(len(bit_stuffed_crc))
            
            # print("perturbed bits:", perturbed_image[b, :, row, :])
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def fgsm_attack_valid(image, data_grad,ep,perturbation_type,pack):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    if perturbation_type == "Random":
        mask = generate_multiple_mask_random(image, pack=1) 

    else:
        mask = generate_max_grad_mask(image, data_grad)

    if mask == None:
        return None
    # print_image(mask,1,pack)
    
    sign_data_grad = sign_data_grad * mask

    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + ep * sign_data_grad
    # print_bits_from_image(perturbed_image,mask)
    
    perturbed_image = apply_constraint(image,mask,perturbed_image)
    return perturbed_image

def apply_fgsm_and_check( pack, test_model,target,data_grad,data_denorm,ep,perturbation_type):
    
 
    perturbed_data = fgsm_attack_valid(data_denorm, data_grad,ep,perturbation_type,pack)
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
        return True, pack+1, final_pred, perturbed_data  # Indicate that we need to reapply
    else:
        # print("Perturbation {} successful. No more injection needed, return pack as final perturbation".format(pack))
        return False, pack, final_pred, perturbed_data  # Indicate that we can stop
    

def Attack_procedure(model, test_model, device, test_loader, perturbation_type, ep, max_perturbations):
    pack = 1
    all_preds = []
    all_labels = []
    n_image = 1
    target_model = test_model

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
                
                
                continue_perturbation, pack, final_pred, data_denorm = apply_fgsm_and_check(pack, test_model, target, data_grad, perturbed_data, ep, perturbation_type)
                perturbation_count += 1

            saving_image(data_denorm,n_image)
        else:
            data.requires_grad = True
            model.eval()
            initial_output = model(data)
            final_pred = initial_output.max(1, keepdim=True)[1]
            print("Image no:", n_image, "(Benign image - skipping perturbation)")
            saving_image(data,n_image)
        
        # target_output = target_model(data_denorm)
        # pp = torch.softmax(target_output, dim=1)
        # print(pp)
        # fp = target_output.max(1, keepdim=True)[1]
        # print("target model final preds",fp)
        print("final no of perturbations:", perturbation_count-1) 

        n_image += 1
        all_preds.extend(final_pred.cpu().numpy())
        all_labels.extend(target.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    # with open("perturb attack_preds.txt", "w") as file: file.write("\n".join(all_preds.squeeze()) + "\n")
    return all_preds.squeeze(), all_labels
    
def check_model(test_loader, model_path):


    """
    Load a pre-trained model, evaluate it on the test dataset, and calculate accuracy.

    Parameters:
    - test_loader: DataLoader for the test dataset
    - model_path: Path to the trained model file

    Returns:
    - test_accuracy: Accuracy of the model on the test dataset
    """

    # Load the pre-trained ResNet18 model
    model = models.resnet18(pretrained=True)
    num_classes = 2  # Update based on your specific classification task
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Modify the final layer
    model.load_state_dict(torch.load(model_path))  # Load the trained model weights
    model.eval()  # Set the model to evaluation mode

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move model to the appropriate device

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

    #steps to check before reunning this code.
    #1. save or print perturbed image .
    #2. save or print confusion matrix.
    #3. Decide epsilon, max_perturbation and perturbation_type
    #4. Select the target IDS (test_model_type) and surrogate IDS
    #5. select the data folders, label file and surrogate IDS
    #6. select GPU or CPU in load_model()
     

    #Define paths for dataset and model
    # dataset_dir = './selected_images'
    test_dataset_dir = './selected_images'
    # train_dataset_dir = './selected_images'
    surr_model_path = './Trained_Models/custom_cnn_model_chd_resnet_ 1.pth'
    # surr_model_path = "./Trained_Models/custom_cnn_model_chd_densenet_.pth"
    feedback_model_path =  './Trained_Models/custom_cnn_model_chd_resnet_ 1.pth'
    # feedback_model_path = "./Trained_Models/custom_cnn_model_chd_densenet_.pth"
    test_label_file = "selected_images.txt"
    feedback_model_type = 'resnet'
    surr_model_type='resnet'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_datasets, test_loader = load_dataset(test_dataset_dir,test_label_file,device,is_train=False)
    print("loaded test dataset")
    # image_datasets, train_loader = load_dataset(train_dataset_dir,train_label_file,device,is_train=True)
    # print("loaded training set")
    # image_datasets,test_loader = load_dataset(dataset_dir)
    
    #laod the model
    model, test_model = load_model(image_datasets, surr_model_path,feedback_model_path, feedback_model_type ,surr_model_type)

    
    # Define the parameters
    epsilon = 1
    perturbation_type = "Random"   
   # List of max_perturbations to iterate over
    max_perturbations_list = [5]
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
