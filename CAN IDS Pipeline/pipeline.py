import sys
from dataset_processing import *
from datasets import CANBusDataset, FeatureExtractedDataset
import config
import os
from sklearn.model_selection import train_test_split

def split_and_store_data(input_dir, test_size=0.2):
    input_dir = os.path.join(input_dir,"modified_dataset")
    train_dir = os.path.join(input_dir,"Train")
    test_dir = os.path.join(input_dir,"Test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    for file in csv_files:
        file_path = os.path.join(input_dir, file)
 
        df = pd.read_csv(file_path)
        
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

        train_df.to_csv(os.path.join(train_dir, file), index=False)
        test_df.to_csv(os.path.join(test_dir, file), index=False)

        print(f"Processed: {file}")

def helper(dataset_path):
    can_datasets = []
    file_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(('.csv', '.txt','.log'))]
    for file_path in file_paths:
        can_datasets.append(CANBusDataset(file_path))
    split_and_store_data(dataset_path)
    # Create a FeatureExtractor
    feature_extractor = FeatureExtractor(
    dataset=can_datasets,
    )
    # Extract features
    feature_extracted_dataset = feature_extractor.extract_features()

    #save_dir = config.FEATURE_SAVE_DIR
    #feature_extracted_dataset.save_dataset(save_dir)

    # Load the dataset
    #loaded_feature_dataset = FeatureExtractedDataset.load_dataset(save_dir)


def main_menu():
    while(True):
        folder_path = input("Dataset folder path : ")
        dir_path = os.path.dirname(os.path.abspath(__file__))
        dir_path = os.path.join(dir_path,"datasets")
        dataset_path = os.path.join(dir_path, folder_path) + "/"
        helper(dataset_path)
            
        
if __name__ == "__main__":
    main_menu()

    # Verify the loaded dataset
    #print(f"Loaded dataset size: {len(loaded_feature_dataset)}")
    #print(f"Features shape: {loaded_feature_dataset[0][0].shape}")
    #print(f"First label: {loaded_feature_dataset[0][1]}")
    #print(f"Feature names: {loaded_feature_dataset.get_feature_names()}")
    #print(f"Label mapping: {loaded_feature_dataset.get_label_mapping()}")

    