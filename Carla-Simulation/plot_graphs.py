import matplotlib.pyplot as plt
import re

import numpy as np

def plot_diff(timestamps, filename):
   
    arr = []

    for i in range(0, len(timestamps)-1):
        if timestamps[i+1] >= 10:
            break
        diff = (timestamps[i+1] - timestamps[i])*1000
        arr.append(diff)


    # Plot the values
    plt.figure(figsize=(20, 6))
    plt.plot(arr, marker='o', linestyle='-', color='b')

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('Time Difference (Milliseconds)')
    plt.title('Plot of Time Differences in Milliseconds')

    # Show the plot
    plt.grid(True)
    plt.savefig(filename)  # Save the plot as a PNG file

def plot_graph(values, filename):
    arr = []
    # Convert values to milliseconds
    for i in range(0, len(values)):
        arr.append(float(values[i])*1000)  # Convert to milliseconds

    # Plot the values
    plt.figure(figsize=(20, 6))
    plt.plot(arr, marker='o', linestyle='-', color='b')

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('Time Difference (Milliseconds)')
    plt.title('Plot of Time Differences in Milliseconds')

    # Show the plot
    plt.grid(True)
    plt.savefig(filename)  # Save the plot as a PNG file

def plot_x_time(location, timestamps, filename):
    # Regex pattern to extract X coordinate
    pattern = r"x=([-+]?\d*\.\d+)"

    x_values = []
    for loc in location:
        x_values.append(float(re.search(pattern, str(loc)).group(1)))
    
    # Plot X coordinate vs. Time
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, x_values, marker='o', linestyle='-', color='b', label='X Coordinate')

    # Labels and title
    plt.xlabel("Time (seconds)")
    plt.ylabel("X Coordinate")
    plt.title("X Coordinate vs. Time")
    plt.grid(True)
    plt.savefig(filename)

def plot_y_time(location, timestamps, filename):
    pattern = r"y=([-+]?\d*\.\d+)"

    y_values = []
    for loc in location:
        y_values.append(float(re.search(pattern, str(loc)).group(1)))

    # Plot Y coordinate vs. Time
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, y_values, marker='o', linestyle='-', color='b', label='Y Coordinate')

    # Labels and title
    plt.xlabel("Time (seconds)")
    plt.ylabel("Y Coordinate")
    plt.title("Y Coordinate vs. Time")
    plt.grid(True)
    plt.savefig(filename)
   
# def plot_vc_time(control_objs, timestamps, filename_throttle, filename_steer, filename_brake):
    
#     pattern = r"throttle=([-+]?\d*\.\d+).*?steer=([-+]?\d*\.\d+).*?brake=([-+]?\d*\.\d+)"

#     throttle_values = []
#     steer_values = []
#     brake_values = []

#     for control in control_objs:
#         throttle_values.append(float(re.search(pattern, str(control)).group(1)))
#         steer_values.append(float(re.search(pattern, str(control)).group(2)))
#         brake_values.append(float(re.search(pattern, str(control)).group(3)))

#     # Plot throttle, steer, and brake vs. Time
#     plt.figure(figsize=(10, 6))
#     plt.plot(timestamps, throttle_values, marker='o', linestyle='-', color='b', label='Throttle')
#     # Labels and title
#     plt.xlabel("Time (seconds)")
#     plt.ylabel("Throttle")
#     plt.title("Throttle vs. Time")
#     plt.grid(True)
#     plt.savefig(filename_throttle)

#     plt.figure(figsize=(10, 6))
#     plt.plot(timestamps, steer_values, marker='o', linestyle='-', color='b', label='Steer')
#     # Labels and title
#     plt.xlabel("Time (seconds)")
#     plt.ylabel("Steer")
#     plt.title("Steer vs. Time")
#     plt.grid(True)
#     plt.savefig(filename_steer)

#     plt.figure(figsize=(10, 6))
#     plt.plot(timestamps, brake_values, marker='o', linestyle='-', color='b', label='Brake')
#     # Labels and title
#     plt.xlabel("Time (seconds)")
#     plt.ylabel("Brake")
#     plt.title("Brake vs. Time")
#     plt.grid(True)
#     plt.savefig(filename_brake)

def plot_vc_time(control_objs, timestamps, filename_throttle, filename_steer, filename_brake):
    
    pattern = r"throttle=([-+]?\d*\.\d+).*?steer=([-+]?\d*\.\d+).*?brake=([-+]?\d*\.\d+)"

    throttle_values = []
    steer_values = []
    brake_values = []
    filtered_timestamps = []

    for control, time in zip(control_objs, timestamps):
        if 20 <= time <= 30:  # Filtering timestamps between 20s and 30s
            match = re.search(pattern, str(control))
            if match:
                throttle_values.append(float(match.group(1)))
                steer_values.append(float(match.group(2)))
                brake_values.append(float(match.group(3)))
                filtered_timestamps.append(time)

    # Plot throttle, steer, and brake vs. Time (Filtered)
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_timestamps, throttle_values, marker='o', linestyle='-', color='b', label='Throttle')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Throttle")
    plt.title("Throttle vs. Time")
    plt.grid(True)
    plt.savefig(filename_throttle)

    plt.figure(figsize=(10, 6))
    plt.plot(filtered_timestamps, steer_values, marker='o', linestyle='-', color='r', label='Steer')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Steer")
    plt.title("Steer vs. Time")
    plt.grid(True)
    plt.savefig(filename_steer)

    plt.figure(figsize=(10, 6))
    plt.plot(filtered_timestamps, brake_values, marker='o', linestyle='-', color='g', label='Brake')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Brake")
    plt.title("Brake vs. Time")
    plt.grid(True)
    plt.savefig(filename_brake)
    
def plot_path_diff(location_1, location_2, timestamps, filename):
    pattern = r"x=([-+]?\d*\.\d+).*?y=([-+]?\d*\.\d+)"

    x_values_1 = []
    y_values_1 = []
    for loc in location_1:
        x_values_1.append(float(re.search(pattern, str(loc)).group(1)))
        y_values_1.append(float(re.search(pattern, str(loc)).group(2)))

    x_values_2 = []
    y_values_2 = []
    for loc in location_2:
        x_values_2.append(float(re.search(pattern, str(loc)).group(1)))
        y_values_2.append(float(re.search(pattern, str(loc)).group(2)))

    # Calculate differences in x and y coordinates
    x_diff = []
    y_diff = []
    for x1, y1, x2, y2 in zip(x_values_1, y_values_1, x_values_2, y_values_2):
        x_diff.append(abs(x2 - x1))
        y_diff.append(abs(y2 - y1))

    # Plot x_diff vs time
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps[:len(x_diff)], x_diff, linestyle='-', marker='o', color='r', label='X Coordinate Difference between Car 1 and Car2')
    
    # Plot y_diff vs time
    plt.plot(timestamps[:len(y_diff)], y_diff, linestyle='-', marker='o', color='b', label='Y Coordinate Difference between Car 1 and Car 2')

    # Labels and title
    plt.xlabel("Time (s)")
    plt.ylabel("Coordinate Difference")
    plt.title("X and Y Coordinate Differences vs Time")
    plt.grid(True)
    plt.legend()
    
    # Save the plot to the file
    plt.savefig(filename)

# def plot_euclid_diff(location_1, location_2, timestamps, filename):
#     pattern = r"x=([-+]?\d*\.\d+).*?y=([-+]?\d*\.\d+)"

#     x_values_1 = []
#     y_values_1 = []
#     for loc in location_1:
#         x_values_1.append(float(re.search(pattern, str(loc)).group(1)))
#         y_values_1.append(float(re.search(pattern, str(loc)).group(2)))

#     x_values_2 = []
#     y_values_2 = []
#     for loc in location_2:
#         x_values_2.append(float(re.search(pattern, str(loc)).group(1)))
#         y_values_2.append(float(re.search(pattern, str(loc)).group(2)))

#     # Calculate Euclidean distances between corresponding points
#     distances = []
#     for x1, y1, x2, y2 in zip(x_values_1, y_values_1, x_values_2, y_values_2):
#         distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
#         distances.append(distance)

#     # Plot Euclidean Distance vs Time
#     plt.figure(figsize=(10, 6))
#     plt.plot(timestamps[:len(distances)], distances, linestyle='-', marker='o', color='b', label='Euclidean Distance')

#     # Labels and title
#     plt.xlabel("Time (s)")
#     plt.ylabel("Euclidean Distance")
#     plt.title("Euclidean Distance vs Time")
#     plt.grid(True)
#     plt.legend()
    
#     # Save the plot to the file
#     plt.savefig(filename)

def plot_euclid_diff(location_1, location_2, timestamps, filename):
    pattern = r"x=([-+]?\d*\.\d+).*?y=([-+]?\d*\.\d+)"
    
    x_values_1, y_values_1 = [], []
    x_values_2, y_values_2 = [], []
    filtered_timestamps = []

    # Filtering data based on timestamps
    for loc1, loc2, time in zip(location_1, location_2, timestamps):
        if 0 <= time <= 30:
            match1 = re.search(pattern, str(loc1))
            match2 = re.search(pattern, str(loc2))
            if match1 and match2:
                x_values_1.append(float(match1.group(1)))
                y_values_1.append(float(match1.group(2)))
                x_values_2.append(float(match2.group(1)))
                y_values_2.append(float(match2.group(2)))
                filtered_timestamps.append(time)

    # Calculate Euclidean distances between corresponding points
    distances = [
        np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        for x1, y1, x2, y2 in zip(x_values_1, y_values_1, x_values_2, y_values_2)
    ]

    # Plot Euclidean Distance vs Time (Filtered)
    plt.figure(figsize=(12, 8))
    plt.plot(filtered_timestamps[:len(distances)], distances, 'g-', label='Benign Euclidean Distance', 
             marker='o', markersize=5, linewidth=3, alpha=0.8)

    # Labels and title
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Euclidean Distance", fontsize=14)
    plt.title("Euclidean Distance vs Time", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)

    # Save the plot to the file
    plt.savefig(filename,dpi=300)

def plot_euclid_diff_single_function(attack_location_1, attack_location_2, attack_timestamps, benign_log_dir,filename, time_range=(0, 30)):
    # Regular expression pattern to extract x and y coordinates
    pattern = r"x=([-+]?\d*\.\d+).*?y=([-+]?\d*\.\d+)"

    # Read and process benign log files
    try:
        with open(f"{benign_log_dir}/gen_coord_1.log", "r") as f1, \
             open(f"{benign_log_dir}/gen_coord_2.log", "r") as f2, \
             open(f"{benign_log_dir}/timestamps_gen.log", "r") as f3:
            
            benign_location_1 = f1.readlines()
            benign_location_2 = f2.readlines()
            benign_timestamps = [float(line.strip()) for line in f3.readlines()]
    except Exception as e:
        print(f"Error reading benign log files from {benign_log_dir}: {e}")
        return

    # # Read and process attack log files
    # try:
    #     with open(f"{attack_log_dir}/gen_coord_1.log", "r") as f1, \
    #          open(f"{attack_log_dir}/gen_coord_2.log", "r") as f2, \
    #          open(f"{attack_log_dir}/timestamps_gen.log", "r") as f3:
            
    #         attack_location_1 = f1.readlines()
    #         attack_location_2 = f2.readlines()
    #         attack_timestamps = [float(line.strip()) for line in f3.readlines()]
    # except Exception as e:
    #     print(f"Error reading attack log files from {attack_log_dir}: {e}")
    #     return

    # print('Benign: ',benign_timestamps)
    # print()
    # print('Attack: ',attack_timestamps)

    # Filter benign data based on time range
    b_x1, b_y1, b_x2, b_y2, b_filtered_timestamps = [], [], [], [], []
    for loc1, loc2, time in zip(benign_location_1, benign_location_2, benign_timestamps):
        if time_range[0] <= time <= time_range[1]:
            match1 = re.search(pattern, str(loc1))
            match2 = re.search(pattern, str(loc2))
            if match1 and match2:
                b_x1.append(float(match1.group(1)))
                b_y1.append(float(match1.group(2)))
                b_x2.append(float(match2.group(1)))
                b_y2.append(float(match2.group(2)))
                b_filtered_timestamps.append(time)

    # Filter attack data based on time range
    a_x1, a_y1, a_x2, a_y2, a_filtered_timestamps = [], [], [], [], []
    for loc1, loc2, time in zip(attack_location_1, attack_location_2, attack_timestamps):
        if time_range[0] <= time <= time_range[1]:
            match1 = re.search(pattern, str(loc1))
            match2 = re.search(pattern, str(loc2))
            if match1 and match2:
                a_x1.append(float(match1.group(1)))
                a_y1.append(float(match1.group(2)))
                a_x2.append(float(match2.group(1)))
                a_y2.append(float(match2.group(2)))
                a_filtered_timestamps.append(time)

    

    # Calculate Euclidean distances for benign and attack data
    b_distances = np.sqrt((np.array(b_x2) - np.array(b_x1))**2 + (np.array(b_y2) - np.array(b_y1))**2)
    a_distances = np.sqrt((np.array(a_x2) - np.array(a_x1))**2 + (np.array(a_y2) - np.array(a_y1))**2)

    # Plot Euclidean Distance vs Time (Filtered)
    plt.figure(figsize=(12, 8))

    # Plotting benign data
    plt.plot(b_filtered_timestamps[:len(b_distances)], b_distances, 'g-', label='Benign Euclidean Distance', 
             marker='o', markersize=5, linewidth=3, alpha=0.8)

    # Plotting attack data
    plt.plot(a_filtered_timestamps[:len(a_distances)], a_distances, 'r-', label='Attack Euclidean Distance', 
             marker='o', markersize=5, alpha=0.15, linewidth=3)

    # Labels and title
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Euclidean Distance", fontsize=14)
    plt.title("Euclidean Distance Comparison (Benign vs Attack)", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)

    # Save the plot to the file
    plt.savefig(filename, dpi=300)


# def plot_trajectories_with_customization(location_1, location_2, timestamps,filename):
#     pattern = r"x=([-+]?\d*\.\d+).*?y=([-+]?\d*\.\d+)"

#     x_values_1 = []
#     y_values_1 = []
#     for loc in location_1:
#         x_values_1.append(float(re.search(pattern, str(loc)).group(1)))
#         y_values_1.append(float(re.search(pattern, str(loc)).group(2)))

#     x_values_2 = []
#     y_values_2 = []
#     for loc in location_2:
#         x_values_2.append(float(re.search(pattern, str(loc)).group(1)))
#         y_values_2.append(float(re.search(pattern, str(loc)).group(2)))

#     # Convert lists to numpy arrays for easier indexing
#     coords1 = np.array([x_values_1, y_values_1]).T
#     coords2 = np.array([x_values_2, y_values_2]).T

#     # Plotting the trajectories with negated y values
#     plt.figure(figsize=(8, 6))
#     plt.plot(coords1[:, 0], coords1[:, 1], 'g-', label='Leading Car Trajectory', marker='o', markersize=1, linewidth=4)
#     plt.plot(coords2[:, 0], coords2[:, 1], 'r-', label='Following Car Trajectory', marker='o', markersize=1, alpha=0.15, linewidth=4)

#     # Hollow start and end coordinate circles for leading trajectory
#     plt.scatter(coords1[0, 0], coords1[0, 1], color='blue', s=100, label='Start Coordinate (Leading)', 
#                 edgecolors='blue', marker='o', facecolors='none', linewidth=2, zorder=5)
#     plt.scatter(coords1[-1, 0], coords1[-1, 1], color='cyan', s=100, label='End Coordinate (Leading)', 
#                 edgecolors='darkcyan', marker='o', facecolors='none', linewidth=2, zorder=5)

#     # Hollow start and end coordinate circles for following trajectory
#     plt.scatter(coords2[0, 0], coords2[0, 1], color='brown', s=100, label='Start Coordinate (Following)', 
#                 edgecolors='brown', marker='o', facecolors='none', linewidth=2, zorder=5)
#     plt.scatter(coords2[-1, 0], coords2[-1, 1], color='magenta', s=100, label='End Coordinate (Following)', 
#                 edgecolors='darkmagenta', marker='o', facecolors='none', linewidth=2, zorder=5)

#     # Adding labels and title
#     plt.xlabel('X Coordinate')
#     plt.ylabel('Y Coordinate')
#     plt.title('Car Trajectories: Leading vs Following')
    
#     # Adding the legend
#     plt.legend()

#     # Removing grid lines and customizing them if necessary
#     plt.grid(True, linestyle=':', linewidth=0.5, color='gray')  # Light gridlines

#     # Save the plot to the file
#     plt.savefig(filename)

def plot_trajectories_with_customization(location_1, location_2, timestamps, filename):
    pattern = r"x=([-+]?\d*\.\d+).*?y=([-+]?\d*\.\d+)"

    x_values_1, y_values_1 = [], []
    x_values_2, y_values_2 = [], []
    filtered_timestamps = []

    # Filter locations based on timestamps between 22.5s and 30s
    for loc1, loc2, time in zip(location_1, location_2, timestamps):
        if 23 <= time <= 30:
            match1 = re.search(pattern, str(loc1))
            match2 = re.search(pattern, str(loc2))
            if match1 and match2:
                x_values_1.append(float(match1.group(1)))
                y_values_1.append(float(match1.group(2)))
                x_values_2.append(float(match2.group(1)))
                y_values_2.append(float(match2.group(2)))
                filtered_timestamps.append(time)

    # Convert lists to numpy arrays for easier indexing
    coords1 = np.array([x_values_1, y_values_1]).T
    coords2 = np.array([x_values_2, y_values_2]).T

    # Plotting the trajectories with filtered data
    plt.figure(figsize=(12, 10))
    plt.plot(coords1[:, 0], coords1[:, 1], 'g-', label='Leading Car Trajectory', marker='o', markersize=1, linewidth=4)
    plt.plot(coords2[:, 0], coords2[:, 1], 'r-', label='Following Car Trajectory', marker='o', markersize=1, alpha=0.15, linewidth=4)

    # Hollow start and end coordinate circles for leading trajectory
    if len(coords1) > 0:
        plt.scatter(coords1[0, 0], coords1[0, 1], color='blue', s=100, label='Start Coordinate (Leading)', 
                    edgecolors='blue', marker='o', facecolors='none', linewidth=2, zorder=5)
        plt.scatter(coords1[-1, 0], coords1[-1, 1], color='cyan', s=100, label='End Coordinate (Leading)', 
                    edgecolors='darkcyan', marker='o', facecolors='none', linewidth=2, zorder=5)

    # Hollow start and end coordinate circles for following trajectory
    if len(coords2) > 0:
        plt.scatter(coords2[0, 0], coords2[0, 1], color='brown', s=100, label='Start Coordinate (Following)', 
                    edgecolors='brown', marker='o', facecolors='none', linewidth=2, zorder=5)
        plt.scatter(coords2[-1, 0], coords2[-1, 1], color='magenta', s=100, label='End Coordinate (Following)', 
                    edgecolors='darkmagenta', marker='o', facecolors='none', linewidth=2, zorder=5)

    # Adding labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Car Trajectories: Leading vs Following')

    # Adding the legend
    plt.legend()

    # Customizing grid
    plt.grid(True, linestyle=':', linewidth=0.5, color='gray')

    # Save the plot
    plt.savefig(filename)

def plot_can_log_diff(filepath, filename):
    timestamp_pattern = r"\((\d+\.\d+)\)" 
    timestamps = []
    with open(filepath, 'r') as file:
        for log in file:
            match = re.search(timestamp_pattern, log)
            if match:
                timestamps.append(float(match.group(1)))
    timestamp_diffs = []
    for i in range(0, len(timestamps) - 2, 3):
        # Group 1: Log 1 -> Log 2
        diff_1 = (timestamps[i + 1] - timestamps[i]) * 1_000_000
        # Group 2: Log 2 -> Log 3
        diff_2 = (timestamps[i + 2] - timestamps[i + 1]) * 1_000_000
        timestamp_diffs.append(diff_1)
        timestamp_diffs.append(diff_2)

    plt.figure(figsize=(20, 6))
    # Plot the timestamp differences
    plt.plot(timestamp_diffs, marker='o', linestyle='-', color='b')
    plt.title('Timestamp Differences')
    plt.xlabel('Index')
    plt.ylabel('Time Difference (µs)')
    plt.grid(True, linestyle=':', linewidth=0.5, color='gray')

    # Save the plot to the file
    plt.savefig(filename)




