# dataset_loader.py

import pandas as pd
from PIL import Image, UnidentifiedImageError
from datasets import Dataset
from torchvision import transforms
import os

def load_and_preprocess_dataset(image_dir, csv_path):
    # Load CSV with the correct delimiter
    df = pd.read_csv(csv_path, delimiter=';')

    # Combine the image paths
    df['image'] = df['image_id'].apply(lambda x: f"{image_dir}/{x}")

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
    def preprocess(example):
        try:
            image = Image.open(example['image']).convert("RGB")
            example['pixel_values'] = transforms.ToTensor()(image)  # Ensure it's always a tensor
        except (UnidentifiedImageError, OSError) as e:
            print(f"Error processing image {example['image']}: {e}")
            example['pixel_values'] = None  # Mark as invalid
        return example

    dataset = dataset.map(preprocess)

    # Filter out invalid examples
    dataset = dataset.filter(lambda example: example['pixel_values'] is not None)

    return dataset
