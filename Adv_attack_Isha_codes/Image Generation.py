import json
from PIL import Image
import os

def calculate_interframe_bits_new(frame, timestamp_difference, data_rate, i):
    """
    Calculating the number of interframe bits based on frame parameters and timestamp difference.

    Args:
        frame (str): Binary representation of the frame.
        timestamp_difference (float): Time difference between current and previous frames.
        data_rate (int): Data rate of the CAN bus (bits per second).
        i (int): Index of the current frame.

    Returns:
        str: Binary representation of interframe bits.

    """

    # Calculating the length of the frame in bits
    length_of_frame = len(frame)

    # Calculating the duration of the frame in seconds
    frame_duration = length_of_frame / data_rate

    # Calculating the interframe time (time gap between current and previous frames)
    # interframe_time = timestamp_difference - frame_duration

    # Calculating the number of interframe bits based on the interframe time and data rate
    # interframe_bits = int(data_rate * interframe_time)
    # print(interframe_time)
    # print("idle time length",interframe_bits)
    # Generating a string of '2's representing the interframe bits
    interframe_bits = round(timestamp_difference * data_rate)
    # print("idle time length",interframe_bits)
    return '2' * interframe_bits

def make_image_array(data_array, frame_type, anchor, data_rate):
    """
    Generate binary image arrays based on CAN data.

    Args:
        data_array (list): List of lists containing timestamp and converted binary data.
        frame_type (list): List containing frame types (0 for benign frames, 1 for attack frames).
        anchor (list): List containing unique converted binary data string for anchor frames.
        data_rate (int): Data rate of the CAN bus (bits per second).

    Returns:
        tuple: A tuple containing three elements:
            - binary_matrix (list): List of binary image arrays.
            - y1 (list): List of labels indicating if the frame is an attack frame (1) or not (0).
            - stats (list): List of dictionaries with counts of benign frames, attack frames, and the ratio of attack frames to total frames for each image.
    """

    # Initialising empty lists and variables

    # List to store binary image arrays
    binary_matrix = []

    # List to store labels indicating majority frames
    y1 = []

    # List to store statistics
    stats = []

    # Initialising a 128x128 matrix with '0's
    a = [['0' for _ in range(128)] for _ in range(128)]

    # Initialising index for iterating over data_array
    i = 0

    # Initialising row for the matrix
    x = 0

    # Initialising column for the matrix
    y = 0

    # Initialising flag to distinguish between benign frames and attack frame
    flag = 0

    # Initialising counters for benign and attack frames
    benign_count = 0
    attack_count = 0

    # Initialising counter for total frames in an image
    count = 0

    #image fill fraction to continue anchor logic
    img_fraction=127/4

    anchor_1 = []
    anchor_2 = []
    anchor_3 = []
    anchor_4 = []

    # Iterating over each entry in data_array
    while i < len(data_array):
        # Extracting binary string representation of the frame
        bin_str = data_array[i][1]

        # Incrementing frame count
        count += 1

        # Setting the flag if the frame is a data frame
        if frame_type[i] == 1:
            flag += 1
            attack_count += 1
        else:
            benign_count += 1
        
        #row< 31
        if bin_str in anchor and (x>0 and x < (int(img_fraction*1))):
            while x < (int(img_fraction*1)):
                while y>=0:
                    a[x][y]='4'
                    if y==127:
                        y=0
                        break
                    y+=1
                x+=1
            anchor_1.append(len(binary_matrix)+1)

            # Iterating over each bit in the binary string
            for bit in bin_str:
                # Setting pixel in the matrix to the corresponding bit
                a[x][y] = bit

                # Moving to the next column
                if y == 127:
                    x += 1
                    y = 0
                else:
                    y += 1

            if i < len(data_array) - 1:
                # Calculate interframe bits and add them to the matrix
                timestamp_difference = data_array[i + 1][0] - data_array[i][0]
                # print("Tdiff 1st segment",timestamp_difference)
                interframe_bits = calculate_interframe_bits_new(data_array[i], timestamp_difference, data_rate, i)
                for bit in interframe_bits:
                    a[x][y] = bit
                    if y == 127:
                        x += 1
                        y = 0
                    else:
                        y += 1

                    if x == 128:
                        # Appending matrix to binary_matrix as an image has been generated
                        binary_matrix.append(a)
                        # Appending label indicating frame type to y1
                        y1.append(1 if flag >= (1) else 0)

                        # Appending statistics for the image
                        stats.append({
                            'benign_count': benign_count,
                            'attack_count': attack_count,
                            'attack_ratio': attack_count / count
                        })

                        # Resetting flag, counts, and count
                        flag = 0
                        benign_count = 0
                        attack_count = 0
                        count = 0

                        # Resetting matrix and row, column indexes
                        a = [['0' for _ in range(128)] for _ in range(128)]
                        x = 0
                        y = 0

                        i += 1
                        break

            # Adding row completion bits to start the next frame in a new row
            while y < 128 and y > 0:
                a[x][y] = '3'
                if y == 127:
                    x += 1
                    if x == 128:
                        # Appending matrix to binary_matrix as an image has been generated
                        binary_matrix.append(a)

                        # Appending label indicating frame type to y1
                        y1.append(1 if flag >= (1) else 0)

                        # Appending statistics for the image
                        stats.append({
                            'benign_count': benign_count,
                            'attack_count': attack_count,
                            'attack_ratio': attack_count / count
                        })

                        # Resetting flag, counts, and count
                        flag = 0
                        benign_count = 0
                        attack_count = 0
                        count = 0

                        # Reset matrix and row, column indexes
                        a = [['0' for _ in range(128)] for _ in range(128)]
                        x = 0
                        y = 0
                        break
                    y = 0
                    break
                y += 1
            i += 1

        # 31<row<63
        elif bin_str in anchor and (x>(int(img_fraction*1) and x < (int(img_fraction*2)))):
            while x < (int(img_fraction*2)):
                while y>=0:
                    a[x][y]='4'
                    if y==127:
                        y=0
                        break
                    y+=1
                x+=1
            anchor_2.append(len(binary_matrix)+1)
            # Iterating over each bit in the binary string
            for bit in bin_str:
                # Setting pixel in the matrix to the corresponding bit
                a[x][y] = bit

                # Moving to the next column
                if y == 127:
                    x += 1
                    y = 0
                else:
                    y += 1

            if i < len(data_array) - 1:
                # Calculate interframe bits and add them to the matrix
                timestamp_difference = data_array[i + 1][0] - data_array[i][0]
                # print("Tdiff 2nd segment",timestamp_difference)
                interframe_bits = calculate_interframe_bits_new(data_array[i], timestamp_difference, data_rate, i)
                for bit in interframe_bits:
                    a[x][y] = bit
                    if y == 127:
                        x += 1
                        y = 0
                    else:
                        y += 1

                    if x == 128:
                        # Appending matrix to binary_matrix as an image has been generated
                        binary_matrix.append(a)

                        # Appending label indicating frame type to y1
                        y1.append(1 if flag >= (1) else 0)

                        # Appending statistics for the image
                        stats.append({
                            'benign_count': benign_count,
                            'attack_count': attack_count,
                            'attack_ratio': attack_count / count
                        })

                        # Resetting flag, counts, and count
                        flag = 0
                        benign_count = 0
                        attack_count = 0
                        count = 0

                        # Resetting matrix and row, column indexes
                        a = [['0' for _ in range(128)] for _ in range(128)]
                        x = 0
                        y = 0

                        i += 1
                        break

            # Adding row completion bits to start the next frame in a new row
            while y < 128 and y > 0:
                a[x][y] = '3'
                if y == 127:
                    x += 1
                    if x == 128:
                        # Appending matrix to binary_matrix as an image has been generated
                        binary_matrix.append(a)

                        # Appending label indicating frame type to y1
                        y1.append(1 if flag >= (1) else 0)

                        # Appending statistics for the image
                        stats.append({
                            'benign_count': benign_count,
                            'attack_count': attack_count,
                            'attack_ratio': attack_count / count
                        })

                        # Resetting flag, counts, and count
                        flag = 0
                        benign_count = 0
                        attack_count = 0
                        count = 0

                        # Reset matrix and row, column indexes
                        a = [['0' for _ in range(128)] for _ in range(128)]
                        x = 0
                        y = 0
                        break
                    y = 0
                    break
                y += 1
            i += 1

        # 63<row<95
        elif bin_str in anchor and (x>(int(img_fraction*2) and x < (int(img_fraction*3)))):
            while x < (int(img_fraction*3)):
                while y>=0:
                    a[x][y]='4'
                    if y==127:
                        y=0
                        break
                    y+=1
                x+=1
            anchor_3.append(len(binary_matrix)+1)
            # Iterating over each bit in the binary string
            for bit in bin_str:
                # Setting pixel in the matrix to the corresponding bit
                a[x][y] = bit

                # Moving to the next column
                if y == 127:
                    x += 1
                    y = 0
                else:
                    y += 1

            if i < len(data_array) - 1:
                # Calculate interframe bits and add them to the matrix
                timestamp_difference = data_array[i + 1][0] - data_array[i][0]
                # print("Tdiff 3rd segment",timestamp_difference)
                interframe_bits = calculate_interframe_bits_new(data_array[i], timestamp_difference, data_rate, i)
                for bit in interframe_bits:
                    a[x][y] = bit
                    if y == 127:
                        x += 1
                        y = 0
                    else:
                        y += 1

                    if x == 128:
                        # Appending matrix to binary_matrix as an image has been generated
                        binary_matrix.append(a)

                        # Appending label indicating frame type to y1
                        y1.append(1 if flag >= (1) else 0)

                        # Appending statistics for the image
                        stats.append({
                            'benign_count': benign_count,
                            'attack_count': attack_count,
                            'attack_ratio': attack_count / count
                        })

                        # Resetting flag, counts, and count
                        flag = 0
                        benign_count = 0
                        attack_count = 0
                        count = 0

                        # Resetting matrix and row, column indexes
                        a = [['0' for _ in range(128)] for _ in range(128)]
                        x = 0
                        y = 0

                        i += 1
                        break

            # Adding row completion bits to start the next frame in a new row
            while y < 128 and y > 0:
                a[x][y] = '3'
                if y == 127:
                    x += 1
                    if x == 128:
                        # Appending matrix to binary_matrix as an image has been generated
                        binary_matrix.append(a)

                        # Appending label indicating frame type to y1
                        y1.append(1 if flag >= (1) else 0)

                        # Appending statistics for the image
                        stats.append({
                            'benign_count': benign_count,
                            'attack_count': attack_count,
                            'attack_ratio': attack_count / count
                        })

                        # Resetting flag, counts, and count
                        flag = 0
                        benign_count = 0
                        attack_count = 0
                        count = 0

                        # Reset matrix and row, column indexes
                        a = [['0' for _ in range(128)] for _ in range(128)]
                        x = 0
                        y = 0
                        break
                    y = 0
                    break
                y += 1
            i += 1

        # 95<row<127
        elif bin_str in anchor and (x>(int(img_fraction*3) and x < (int(img_fraction*4)))):
            while x < (int(img_fraction*4)):
                while y>=0:
                    a[x][y]='4'
                    if y==127:
                        y=0
                        break
                    y+=1
                x+=1
            anchor_4.append(len(binary_matrix)+1)
            # Iterating over each bit in the binary string
            for bit in bin_str:
                # Setting pixel in the matrix to the corresponding bit
                a[x][y] = bit

                # Moving to the next column
                if y == 127:
                    # Appending matrix to binary_matrix as an image has been generated
                    binary_matrix.append(a)

                    # Appending label indicating frame type to y1
                    y1.append(1 if flag >= (1) else 0)

                    # Appending statistics for the image
                    stats.append({
                        'benign_count': benign_count,
                        'attack_count': attack_count,
                        'attack_ratio': attack_count / count
                    })

                    # Resetting flag, counts, and count
                    flag = 0
                    benign_count = 0
                    attack_count = 0
                    count = 0

                    # Reset matrix and row, column indexes
                    a = [['0' for _ in range(128)] for _ in range(128)]
                    x = 0
                    y = 0
                    break
                else:
                    y += 1

            if (i < len(data_array) - 1) and y!=0:
                # Calculate interframe bits and add them to the matrix
                timestamp_difference = data_array[i + 1][0] - data_array[i][0]
                # print("Tdiff 4th segment",timestamp_difference)
                interframe_bits = calculate_interframe_bits_new(data_array[i], timestamp_difference, data_rate, i)
                for bit in interframe_bits:
                    a[x][y] = bit
                    if y == 127:
                        # Appending matrix to binary_matrix as an image has been generated
                        binary_matrix.append(a)

                        # Appending label indicating frame type to y1
                        y1.append(1 if flag >= (1) else 0)

                        # Appending statistics for the image
                        stats.append({
                            'benign_count': benign_count,
                            'attack_count': attack_count,
                            'attack_ratio': attack_count / count
                        })

                        # Resetting flag, counts, and count
                        flag = 0
                        benign_count = 0
                        attack_count = 0
                        count = 0

                        # Resetting matrix and row, column indexes
                        a = [['0' for _ in range(128)] for _ in range(128)]
                        x = 0
                        y = 0
                        break
                    else:
                        y += 1
            i += 1


        # Frame is not an anchor frame
        else:
            # Iterating over each bit in the binary string
            for bit in bin_str:
                # Setting pixel in the matrix to the corresponding bit
                a[x][y] = bit

                # Moving to the next column
                if y == 127:
                    x += 1
                    y = 0
                else:
                    y += 1

                # Checking if end of the matrix is reached
                if x == 128:
                    # Append matrix to binary_matrix
                    binary_matrix.append(a)

                    # Append label indicating majority frame to y1
                    y1.append(1 if flag >= (1) else 0)

                    # Append statistics for the image
                    stats.append({
                        'benign_count': benign_count,
                        'attack_count': attack_count,
                        'attack_ratio': attack_count / count
                    })

                    # Reset flag, counts, and count
                    flag = 0
                    benign_count = 0
                    attack_count = 0
                    count = 0

                    # Resetting matrix and row, column indexes
                    a = [['0' for _ in range(128)] for _ in range(128)]
                    x = 0
                    y = 0

                    i += 1
                    break

            # Check if there are more frames
            if (i < len(data_array) - 1) and (x or y):
                # Calculate interframe bits and add them to the matrix
                timestamp_difference = data_array[i + 1][0] - data_array[i][0]
                # print("Tdiff 5th segment",timestamp_difference)
                interframe_bits = calculate_interframe_bits_new(data_array[i], timestamp_difference, data_rate, i)
                for bit in interframe_bits:
                    a[x][y] = bit
                    if y == 127:
                        x += 1
                        y = 0
                    else:
                        y += 1

                    if x == 128:
                        # Appending matrix to binary_matrix as an image has been generated
                        binary_matrix.append(a)

                        # Appending label indicating frame type to y1
                        y1.append(1 if flag >= (1) else 0)

                        # Appending statistics for the image
                        stats.append({
                            'benign_count': benign_count,
                            'attack_count': attack_count,
                            'attack_ratio': attack_count / count
                        })

                        # Resetting flag, counts, and count
                        flag = 0
                        benign_count = 0
                        attack_count = 0
                        count = 0

                        # Resetting matrix and row, column indexes
                        a = [['0' for _ in range(128)] for _ in range(128)]
                        x = 0
                        y = 0

                        i += 1
                        break

            # Adding row completion bits to start the next frame in a new row
            while (y < 128 and y > 0) and (x or y):
                a[x][y] = '3'
                if y == 127:
                    x += 1
                    if x == 128:
                        # Appending matrix to binary_matrix as an image has been generated
                        binary_matrix.append(a)

                        # Appending label indicating frame type to y1
                        y1.append(1 if flag >= (1) else 0)

                        # Appending statistics for the image
                        stats.append({
                            'benign_count': benign_count,
                            'attack_count': attack_count,
                            'attack_ratio': attack_count / count
                        })

                        # Resetting flag, counts, and count
                        flag = 0
                        benign_count = 0
                        attack_count = 0
                        count = 0

                        # Reset matrix and row, column indexes
                        a = [['0' for _ in range(128)] for _ in range(128)]
                        x = 0
                        y = 0
                        break
                    y = 0
                    break
                y += 1
            
        i += 1

    return binary_matrix, y1, stats, anchor_1,anchor_2,anchor_3,anchor_4

def image_generation(binary_matrix, y1):    
    # Define a color mapping dictionary to map binary values to colors
    color_mapping = {
        '4': (255, 255, 0),  # Yellow for anchor frames
        '3': (255, 0, 0),    # Red for row completion bits
        '2': (0, 255, 0),    # Green for interframe bits
        '1': (255, 255, 255),# White for data bits
        '0': (0, 0, 0)       # Black for empty bits
    }

    # Specifying the base output folder
    base_output_folder = r"./DoS_images"
    os.makedirs(base_output_folder, exist_ok=True)

    # Initializing a counter for the image filenames
    count = 1

    # Creating a text file to store the labels
    label_file_path = os.path.join(base_output_folder, 'DoS_labels.txt')
    with open(label_file_path, 'w') as label_file:

        # Iterating through each 2D list in the binary_matrix
        for idx, two_d_list in enumerate(binary_matrix):

            # Create a blank image with the size of 128x128 pixels
            image_size = (128, 128)
            img = Image.new('RGB', image_size)

            # Iterate through each row in the 2D list
            for i, row in enumerate(two_d_list):

                # Iterate through each element in the row
                for j, element in enumerate(row):

                    # Get the color corresponding to the binary value from the color_mapping dictionary
                    color = color_mapping.get(element, (0, 0, 0))  # Default to black if not found

                    # Set the pixel color in the image at the specified (j, i) position
                    img.putpixel((j, i), color)

            # Generating a unique filename for the image
            filename = f'image_{count}.png'

            # Saving the resulting image in the base output folder
            output_path = os.path.join(base_output_folder, filename)
            img.save(output_path)

            # print("Image{} saved".format(count))

            # Write the label to the truth labels file
            label_file.write(f'{filename}: {y1[idx]}\n')

            # Incrementing the counter for the next image filename
            count += 1

def load_json(path):

    with open(path, 'r') as file:
        data = json.load(file)

    data_array = data['data_array']
    frame_type = data['frame_type']
    anchor = data['anchor']

    return data_array, frame_type, anchor


def main():
    
    json_file_path = 'DoS_data_CH.json'
    # Load the JSON file
    data_array, frame_type, anchor = load_json(json_file_path)
    print("json loaded")
    # Formation of Images
    binary_matrix, y1, stats, anchor_1, anchor_2, anchor_3, anchor_4 = make_image_array(data_array, frame_type, anchor, data_rate=500000)  #1024000
    
    image_generation(binary_matrix, y1)
    # print("All images saved in the CHD_images folder")


if __name__ == "__main__":
    main()