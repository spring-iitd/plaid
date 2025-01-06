import pandas as pd
import sys

def process_traffic_logs(mutation_operation,perturbed_file_path,original_file_path, output_file_path):
    """
    Processes CAN traffic log files and adds a Label column based on conditions.

    Parameters:
    - converted_file_path: str, path to the converted traffic file.
    - perturbed_file_path: str, path to the perturbed traffic file.
    - output_file_path: str, path to save the processed output file as an Excel file.
    """
    # Define column names
    columns = ['TS', 'ID', 'DLC', 'Data1', 'Data2', 'Data3', 'Data4', 'Data5', 'Data6', 'Data7', 'Data8']

    # Read the perturbed traffic file
    perturbed_traffic = pd.read_csv(perturbed_file_path, names=columns,dtype=str)
    original_traffic = pd.read_csv(original_file_path, names=columns,dtype=str)

    # Add a 'Label' column to perturbed_traffic with default label 0
    perturbed_traffic['Label'] = 0
    # Label 1: Rows with ID '000'
    perturbed_traffic.loc[perturbed_traffic['ID'] == '000', 'Label'] = 1   
    
    # Add Label column based on ID values
    if mutation_operation == "Injection":
             
        merged = perturbed_traffic.merge(original_traffic, on=['TS', 'ID'], how='left', indicator=True)
        perturbed_traffic.loc[(merged['_merge'] == 'left_only') & (perturbed_traffic['ID'] == '130'), 'Label'] = 2
        perturbed_traffic.to_csv(output_file_path, index=False)
        print(f"Processed data saved to {output_file_path}")

    elif mutation_operation == "Modification":

        # Merge on TS to align rows from both datasets
        merged = perturbed_traffic.merge(original_traffic, on='TS', suffixes=('_perturbed', '_original'))
        # Find rows where the original ID was '000' but the perturbed ID is different
        changed_rows = merged[(merged['ID_original'] == '000') & (merged['ID_perturbed'] != '000')]
        # Update the labels in perturbed_traffic based on matching TS
        perturbed_traffic.loc[perturbed_traffic['TS'].isin(changed_rows['TS']), 'Label'] = 2
        #save the file
        print("\n",perturbed_traffic)
        perturbed_traffic.to_csv(output_file_path, index=False)
        print(f"Processed data saved to {output_file_path}")
    
    elif mutation_operation == "Both":
        # Merge on TS to align rows from both datasets
        merged = perturbed_traffic.merge(original_traffic, on='TS', suffixes=('_perturbed', '_original'))

        # Label 2: a) Rows where original '000' became different but same TS in perturbed traffic
        changed_rows = merged[(merged['ID_original'] == '000') & (merged['ID_perturbed'] != '000')]

        # Update the labels in perturbed_traffic for changed entries based on TS
        perturbed_traffic.loc[perturbed_traffic['TS'].isin(changed_rows['TS']), 'Label'] = 2

        merged = perturbed_traffic.merge(original_traffic, on=['TS', 'ID'], how='left', indicator=True)
        perturbed_traffic.loc[(merged['_merge'] == 'left_only') & (perturbed_traffic['ID'] == '130'), 'Label'] = 2
        
        #save the file
        perturbed_traffic.to_csv(output_file_path, index=False)
        print(f"Processed data saved to {output_file_path}")
    
    else:
        perturbed_traffic.to_csv(output_file_path, index=False)
        print(f"Processed data saved to {output_file_path}")

   


if __name__ == "__main__":
    # File paths
    if len(sys.argv) != 2:
        print("Usage: python file_name.py <PerturbationType>")
        sys.exit(1)

    # Read the perturbation type from the command-line argument
    mutation_operation = sys.argv[1]

    # mutation_operation = "Injection"
    original_file_path = 'original_traffic.txt'  # Update with the correct path if needed
    perturbed_file_path = f'perturbed_traffic_{mutation_operation}.txt'  # Update with the correct path if needed
    output_file = f'Adversarial_traffic_{mutation_operation}.csv'  # Output Excel file
    # Process the files
    process_traffic_logs( mutation_operation,perturbed_file_path, original_file_path,output_file)