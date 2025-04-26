import os
import torch
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DConditionModel
from dataset_loader import load_and_preprocess_dataset

# Configuration
image_dir = "/data/ISIC_2019_Training_Input_100img"  # Replace with your image folder path
csv_path = "/data/ISIC_2019_Training_Input_100img/generated_descriptions_isic_100.csv"  # Replace with your CSV file path
batch_size = 16
num_epochs = 10
learning_rate = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
model_save_dir = "./ddpm_models"  # Directory to save models
os.makedirs(model_save_dir, exist_ok=True)  # Ensure the directory exists

print(f"Using device: {device}")

# Load dataset
print("Loading dataset...")
dataset = load_and_preprocess_dataset(image_dir, csv_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Load CLIP tokenizer and text encoder
print("Loading CLIP tokenizer and text encoder...")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

# Define the DDPM model
print("Initializing DDPM model...")
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
unet = UNet2DConditionModel(
    sample_size=512,  # Image resolution
    in_channels=4,    # Latent space channels
    out_channels=4,
    layers_per_block=2,
    block_out_channels=(128, 256, 512, 512),
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    cross_attention_dim=768
).to(device)

# Optimizer
optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)

# Training loop
print("Starting training...")
for epoch in range(num_epochs):
    unet.train()
    for batch in dataloader:
        # Prepare inputs
        pixel_values = torch.stack(batch["pixel_values"]).to(device)
        text = batch["description"]  # Ensure your CSV has a 'description' column
        text_inputs = tokenizer(text, padding="max_length", max_length=77, return_tensors="pt", truncation=True)
        input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)

        # Encode text
        encoder_hidden_states = text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # Add noise to images
        noise = torch.randn_like(pixel_values).to(device)
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (pixel_values.shape[0],), device=device).long()
        noisy_images = noise_scheduler.add_noise(pixel_values, noise, timesteps)

        # Predict noise
        noise_pred = unet(noisy_images, timesteps, encoder_hidden_states).sample

        # Compute loss
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item()}")

# Determine the next model name
existing_models = [f for f in os.listdir(model_save_dir) if f.startswith("ddpm_model_") and f.endswith(".bin")]
model_numbers = [int(f.split("_")[-1].split(".")[0]) for f in existing_models]
next_model_number = max(model_numbers, default=0) + 1
model_name = f"ddpm_model_{next_model_number}"
model_path = os.path.join(model_save_dir, model_name)

# Save the model
print(f"Saving the model as {model_name}...")
pipeline = DDPMPipeline(unet=unet, scheduler=noise_scheduler)
pipeline.save_pretrained(model_path)

# Save the parameters to a text file
params_file = os.path.join(model_save_dir, f"{model_name}_params.txt")
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