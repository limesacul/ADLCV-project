# train_lora.py

from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer
import torch
from dataset_loader import load_and_preprocess_dataset
import os

def train_lora(
    image_dir, 
    csv_path, 
    output_dir, 
    model_name="runwayml/stable-diffusion-v1-5", 
    rank=4, 
    learning_rate=1e-4, 
    batch_size=1, 
    epochs=5):

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

    # Save LoRA adapters properly
    unet.save_pretrained(output_dir)


if __name__ == "__main__":
    img_dir = "/dtu/blackhole/07/203495/ADLCV-project/data/ISIC_2019_Training_Input"
    prompt_dir = "/dtu/blackhole/07/203495/ADLCV-project/data/ISIC_2019_Training_Input/generated_descriptions_isic_5000.csv"
    train_lora(img_dir, prompt_dir, "image_generation/lora_weights_isic_5000")
