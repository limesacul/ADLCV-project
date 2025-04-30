# train_lora.py

from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer
import torch
from dataset_loader import load_and_preprocess_dataset
import os
from datetime import datetime

def train_lora(
    image_dir, 
    csv_path, 
    output_dir, 
    model_name="runwayml/stable-diffusion-v1-5", 
    rank=4, 
    learning_rate=5e-5, 
    batch_size=2, 
    epochs=10):

    # Load dataset
    dataset = load_and_preprocess_dataset(image_dir, csv_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load pipeline components
    pipeline = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)
    pipeline = pipeline.to(device)
    pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)

    # Prepare U-Net with PEFT/LoRA
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=["to_q", "to_k", "to_v"],
        lora_dropout=0.1,
        bias="none",
        task_type="UNET"
    )

    unet = get_peft_model(pipeline.unet, lora_config)
    unet.enable_gradient_checkpointing()
    unet.train()

    optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            pixel_values = torch.stack([torch.tensor(p) if not isinstance(p, torch.Tensor) else p for p in batch["pixel_values"]]).to(device)
            captions = batch["generated_text"]

            text_inputs = pipeline.tokenizer(
                captions,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)

            encoder_hidden_states = pipeline.text_encoder(**text_inputs).last_hidden_state

            with torch.no_grad():
                latents = pipeline.vae.encode(pixel_values).latent_dist.sample() * 0.18215

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device).long()
            noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)

            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
            loss = loss_fn(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}, Step {i+1}, Loss: {loss.item()}")

    # Create a timestamped model directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lora_model_name = f"lora_model_{timestamp}"
    model_path = os.path.join(output_dir, lora_model_name)
    os.makedirs(model_path, exist_ok=True)

    # Save LoRA adapters properly
    unet.save_pretrained(model_path)
    print("Model saved at:", model_path)

    # Save the parameters to a text file
    params_file = os.path.join(model_path, f"{lora_model_name}_params.txt")
    with open(params_file, "w") as f:
        f.write(f"Model Name: {lora_model_name}\n")
        f.write(f"Image Directory: {image_dir}\n")
        f.write(f"CSV Path: {csv_path}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Number of Epochs: {epochs}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Device: {device}\n")
        f.write(f"LoRA Rank: {rank}\n")
        f.write(f"LoRA Alpha: {rank}\n")
        f.write(f"LoRA Target Modules: {lora_config.target_modules}\n")
        f.write(f"LoRA Dropout: {lora_config.lora_dropout}\n")
        f.write(f"LoRA Bias: {lora_config.bias}\n")
        f.write(f"LoRA Task Type: {lora_config.task_type}\n")

    print("Training parameters saved at:", params_file)


if __name__ == "__main__":
    img_dir = "/dtu/blackhole/07/203495/ADLCV-project/data/ISIC_2019_Training_Input_split/train/images"
    prompt_dir = "/dtu/blackhole/07/203495/ADLCV-project/data/ISIC_2019_Training_Input_split/train_descriptions.csv"
    print("Training images:", img_dir)
    print("Training csv:", prompt_dir)
    train_lora(img_dir, prompt_dir, "/dtu/blackhole/07/203495/ADLCV-project/image_generation/lora_models")
