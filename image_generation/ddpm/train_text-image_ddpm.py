import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DConditionModel
from .dataset_loader import load_and_preprocess_dataset
from datetime import datetime
from torch.amp import autocast, GradScaler

# Configuration
image_dir = "./data/ISIC_2019_Training_Input_128x128"  # Replace with your image folder path
csv_path = "./data/ISIC_2019_Training_Input_128x128/generated_descriptions_isic_5000_128x128.csv"  # Replace with your CSV file path
processed_path = "./data/processed_isic_5000_128x128"
print("Trying noisy_images = noise_scheduler.add_noise(pixel_values, noise, timesteps)")
print("Training images:", image_dir)
print("Training csv:", csv_path)


batch_size = 1
num_epochs = 3
learning_rate = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model_save_dir = "./image_generation/ddpm/ddpm_models"  # Directory to save models
os.makedirs(model_save_dir, exist_ok=True)  # Ensure the directory exists

print(f"Using device: {device}")

# Load dataset
print("Loading dataset...")
dataset = load_and_preprocess_dataset(image_dir, csv_path, processed_path=processed_path)
# Print the number of images found
print(f"Number of images found in the dataset: {len(dataset)}")

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Load CLIP tokenizer and text encoder
print("Loading CLIP tokenizer and text encoder...")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

# Define the DDPM model
print("Initializing DDPM model...")
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
unet = UNet2DConditionModel(
    sample_size=128,  # Image resolution
    in_channels=3,    # Match the input tensor's channels (RGB)
    out_channels=3,   # Match the output tensor's channels (RGB)
    layers_per_block=2,
    block_out_channels=(128, 256, 512, 512),
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    cross_attention_dim=512  # Match the CLIP text encoder's output dimension
).to(device)

# Optimizer
optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)

# Initialize GradScaler
scaler = GradScaler('cuda')

# Training loop
print("Starting training...")
for epoch in range(num_epochs):
    unet.train()
    for batch in dataloader:
        # Prepare inputs
        pixel_values = batch["pixel_values"].to(device)  # Already a tensor
        text = batch["description"]
        text_inputs = tokenizer(text, padding="max_length", max_length=77, return_tensors="pt", truncation=True)
        input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)

        # Encode text
        encoder_hidden_states = text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # Add noise to images
        noise = torch.randn_like(pixel_values).to(device)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (pixel_values.shape[0],), device=device).long()
        noisy_images = noise_scheduler.add_noise(pixel_values, noise, timesteps)

        # Mixed precision training
        with autocast('cuda'):
            # Predict noise
            noise_pred = unet(noisy_images, timesteps, encoder_hidden_states).sample

            # Compute loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

        # Backpropagation with GradScaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item()}")

# Determine the model name using a timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"ddpm_model_{timestamp}"
model_path = os.path.join(model_save_dir, model_name)

print(f"Saving the model as {model_name}...")
pipeline = DDPMPipeline(unet=unet, scheduler=noise_scheduler)
pipeline.save_pretrained(model_path)

# Save the parameters to a text file
params_file = os.path.join(model_path, f"{model_name}_params.txt")
with open(params_file, "w") as f:
    f.write(f"Model Name: {model_name}\n")
    f.write(f"Image Directory: {image_dir}\n")
    f.write(f"CSV Path: {csv_path}\n")
    f.write(f"Batch Size: {batch_size}\n")
    f.write(f"Number of Epochs: {num_epochs}\n")
    f.write(f"Learning Rate: {learning_rate}\n")
    f.write(f"Device: {device}\n")
    f.write(f"Noise Scheduler Timesteps: {noise_scheduler.num_train_timesteps}\n")
    f.write(f"UNet Configuration:\n")
    f.write(f"  Sample Size: {unet.sample_size}\n")
    f.write(f"  In Channels: {unet.in_channels}\n")
    f.write(f"  Out Channels: {unet.out_channels}\n")
    f.write(f"  Layers Per Block: {unet.layers_per_block}\n")
    f.write(f"  Block Out Channels: {unet.block_out_channels}\n")
    f.write(f"  Down Block Types: {unet.down_block_types}\n")
    f.write(f"  Up Block Types: {unet.up_block_types}\n")
    f.write(f"  Cross Attention Dim: {unet.cross_attention_dim}\n")

print("Training complete!")