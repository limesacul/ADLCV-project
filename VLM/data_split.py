import pandas as pd
import numpy as np
import os
from shutil import copyfile

# Set random seed for reproducibility
np.random.seed(42)

# File paths (update these with your actual paths)
image_dir = '/work3/s233349/ADLCV-project/data/ISIC_2019_Training_Input'
metadata_path = '/work3/s233349/ADLCV-project/data/ISIC_2019_Training_Metadata.csv'
ground_truth_path = '/work3/s233349/ADLCV-project/data/ISIC_2019_Training_GroundTruth.csv'
output_dir = '/work3/s233349/ADLCV-project/split_data'

# Create output directories
os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)

# Load data
metadata = pd.read_csv(metadata_path)
ground_truth = pd.read_csv(ground_truth_path)

# Merge metadata and ground truth on 'image' column
merged_data = pd.merge(metadata, ground_truth, on='image')

# Shuffle the dataset
shuffled_data = merged_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into train (80%) and test (20%)
split_idx = int(0.8 * len(shuffled_data))
train_data = shuffled_data.iloc[:split_idx]
test_data = shuffled_data.iloc[split_idx:]

# Function to save split data
def save_split(df, split_name):
    # Save metadata portion
    metadata_cols = ['image', 'age_approx', 'anatom_site_general', 'lesion_id', 'sex']
    df[metadata_cols].to_csv(os.path.join(output_dir, f'{split_name}_metadata.csv'), index=False)
    
    # Save ground truth portion
    gt_cols = ['image', 'MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
    df[gt_cols].to_csv(os.path.join(output_dir, f'{split_name}_ground_truth.csv'), index=False)
    
    # Copy images
    os.makedirs(os.path.join(output_dir, split_name, 'images'), exist_ok=True)
    for img_name in df['image']:
        src = os.path.join(image_dir, f"{img_name}.jpg")
        dst = os.path.join(output_dir, split_name, 'images', f"{img_name}.jpg")
        if os.path.exists(src):  # Check if image exists
            copyfile(src, dst)

# Save both splits
save_split(train_data, 'train')
save_split(test_data, 'test')

# Verification
print("Split completed successfully!")
print(f"Total samples: {len(merged_data)}")
print(f"Training samples: {len(train_data)} ({(len(train_data)/len(merged_data))*100:.1f}%)")
print(f"Test samples: {len(test_data)} ({(len(test_data)/len(merged_data))*100:.1f}%)")
print(f"\nOutput structure in {output_dir}:")
print("train/")
print("├── images/")
print("├── train_ground_truth.csv")
print("└── train_metadata.csv")
print("test/")
print("├── images/")
print("├── test_ground_truth.csv")
print("└── test_metadata.csv")