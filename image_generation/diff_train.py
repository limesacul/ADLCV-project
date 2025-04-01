import os
import pandas as pd
from datasets import load_dataset
from diffusers import StableDiffusionPipeline, StableDiffusionTrainer, DDPMScheduler
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import torch

# Define your dataset class
class TextImageDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, image_size=512):
        self.data = data
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        image_path = item["image_path"]
        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)

        # Tokenize the text
        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        return {
            "pixel_values": image,
            "input_ids": text_inputs.input_ids.squeeze(),
            "attention_mask": text_inputs.attention_mask.squeeze()
        }

# Load your dataset
def load_custom_dataset(data_dir, csv_path):
    """
    Load a dataset of text-image pairs from a directory of images and a CSV file.

    Args:
        data_dir (str): Path to the directory containing images.
        csv_path (str): Path to the CSV file containing image descriptions.

    Returns:
        list: A list of dictionaries with "text" and "image_path" keys.
    """
    # Read the CSV file
    df = pd.read_csv(csv_path, sep="\t")  # Assuming tab-separated values
    dataset = []

    for _, row in df.iterrows():
        image_filename = row["image_filename"]  # Column with image filenames
        text = row["description"]  # Column with textual descriptions
        image_path = os.path.join(data_dir, image_filename)

        # Ensure the image file exists
        if os.path.exists(image_path):
            dataset.append({"text": text, "image_path": image_path})
        else:
            print(f"Warning: Image file {image_path} not found. Skipping.")

    return dataset

# Main training script
def train_model(data_dir, csv_path, model_name="CompVis/stable-diffusion-v1-4", output_dir="./fine_tuned_model", epochs=5, batch_size=4, lr=5e-6):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipeline = StableDiffusionPipeline.from_pretrained(model_name)
    pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)

    # Load the dataset
    dataset = load_custom_dataset(data_dir, csv_path)
    train_dataset = TextImageDataset(dataset, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define the optimizer
    optimizer = torch.optim.AdamW(pipeline.unet.parameters(), lr=lr)

    # Training loop
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(device)

    for epoch in range(epochs):
        pipeline.unet.train()
        for batch in train_dataloader:
            optimizer.zero_grad()

            # Move data to device
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass
            loss = pipeline(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            ).loss

            # Backward pass
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs} completed. Loss: {loss.item()}")

    # Save the fine-tuned model
    pipeline.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    # Replace 'data_dir' and 'csv_path' with the path to your dataset directory and CSV file
    data_dir = "/dtu/blackhole/07/203495/ADLCV-project/data/"
    csv_path = "/dtu/blackhole/07/203495/ADLCV-project/data/"
    train_model(data_dir, csv_path)