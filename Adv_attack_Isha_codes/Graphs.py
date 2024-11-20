
# import pandas as pd
# import matplotlib.pyplot as plt

# # Data from your table
# data = {
#     'Permutation Type': ['Max_grad']*11 + ['Random']*11,
#     'Max No. of Injections': [1, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60]*2,
#     'FN': [8,198,420,663,969,1131,1200,1255,1308,1325,1328,
#            8,160,305,566,951,1134,1234,1291,1332,1343,1355]
# }

# # Create DataFrame
# df = pd.DataFrame(data)

# # Plotting FNs for both permutation types
# fig, ax = plt.subplots(figsize=(10, 6))

# # Plot FN for both permutation types
# for perm_type in df['Permutation Type'].unique():
#     subset = df[df['Permutation Type'] == perm_type]
#     ax.plot(subset['Max No. of Injections'], subset['FN'], label=f'FN - {perm_type}', marker='o')

# # Titles and labels
# ax.set_title('False Negatives (FN) vs Max No. of Injections', fontsize=14)
# ax.set_xlabel('Max No. of Injections', fontsize=12)
# ax.set_ylabel('False Negatives (FN)', fontsize=12)
# ax.grid(True)
# ax.legend()

# # Set y-axis limits with a gap of 100
# y_min = 0  # Start from 0
# y_max = 1500  # Set the maximum to 1500
# ax.set_ylim(y_min, y_max)

# # Set y-axis ticks at intervals of 100
# ax.set_yticks(range(0, y_max + 100, 100))

# plt.tight_layout()
# plt.show()


'''
Code for FPs Vs injections for benign images.
'''

import matplotlib.pyplot as plt

# Data: Number of frame injections vs False Positives
num_injections = [1, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60]
false_positives_mx = [16, 46, 76, 173, 501, 886, 1193, 1342, 1417, 1419, 1419]
false_positives_rd = [30,74,84,124,201,402,583,820,1247,1386,1419]


# Creating the plot
plt.figure(figsize=(10, 6))
plt.plot(num_injections, false_positives_mx, marker='o', color='b', label='False Positives_max_grad')
plt.plot(num_injections, false_positives_rd, marker='o', color='r', label='False Positives_random')


# Adding titles and labels
plt.title('False Positives vs Number of Frame Injections using random', fontsize=14)
plt.xlabel('Max_injection allowed', fontsize=12)
plt.ylabel('Number of False Positives', fontsize=12)
plt.grid(True)

# Highlighting the point where False Positives stabilize
# plt.axvline(x=50, color='r', linestyle='--', label='Maximum False Positives Stabilized')

# # Annotating the threshold where FP stabilizes
# plt.annotate('FP Stabilized', xy=(50, 1419), xytext=(55, 1350),
#              arrowprops=dict(arrowstyle="->", lw=1.5), fontsize=10)

# Show legend
plt.legend()

# Show the plot
plt.show()
