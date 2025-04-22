# dataset_loader.py

import pandas as pd
from PIL import Image, UnidentifiedImageError
from datasets import Dataset
from torchvision import transforms
import os

def load_and_preprocess_dataset(image_dir, csv_path):
    # Load CSV with the correct delimiter and handle quoted column names
    df = pd.read_csv(csv_path, delimiter='\t') #delimiter=';')
    df.columns = df.columns.str.strip('"')  # Remove quotes from column names if present

    # Debugging: Print column names
    print(f"Columns in CSV: {df.columns}")

    # Ensure the column name matches the CSV file
    if 'image_id' not in df.columns:
        raise KeyError("The column 'image_id' is missing from the CSV file. Please check the file format.")

    # Combine the image paths
    df['image'] = df['image_id'].apply(lambda x: f"{image_dir}/{x}.jpg")

    # Check if images exist and filter out missing files
    def check_image_exists(row):
        if not os.path.exists(row['image']):
            print(f"Warning: Image not found at {row['image']}")
            return False
        return True

    df = df[df.apply(check_image_exists, axis=1)]

    # Load dataset with Hugging Face
    dataset = Dataset.from_pandas(df)

    # Preprocessing transforms
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Add image loading and preprocessing with error handling
    def preprocess(batch):
        # Initialize the 'pixel_values' key in the batch
        pixel_values = []
        for example in batch['image']:
            try:
                image = Image.open(example).convert("RGB")
                pixel_values.append(transform(image))  # Apply the transform
            except (UnidentifiedImageError, OSError) as e:
                print(f"Error processing image {example}: {e}")
                pixel_values.append(None)  # Mark as invalid
        batch['pixel_values'] = pixel_values
        return batch

    # Initialize the 'pixel_values' column
    dataset = dataset.map(preprocess, batched=True, batch_size=64)

    # Filter out invalid examples
    dataset = dataset.filter(lambda example: example['pixel_values'] is not None)

    # Print the number of valid images loaded
    print(f"Number of valid images with descriptions loaded: {len(dataset)}")

    return dataset
