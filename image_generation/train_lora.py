# train_lora.py

import torch
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
from dataset_loader import load_and_preprocess_dataset

def train_lora(image_dir, csv_path, output_dir, model_name="runwayml/stable-diffusion-v1-5", rank=4, learning_rate=1e-4, batch_size=1, epochs=5):
    # Load dataset
    dataset = load_and_preprocess_dataset(image_dir, csv_path)

    # Load pre-trained pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)

    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    pipeline = pipeline.to(device)

    # Print the architecture of the unet to identify target modules
    #print("UNet architecture:")
    #print(pipeline.unet)

    # Configure LoRA
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=["to_q", "to_k", "to_v", "proj_in", "proj_out"],  # Supported modules
        lora_dropout=0.1,
        bias="none",
        modules_to_save=[],
    )

    # Prepare model for LoRA
    unet = get_peft_model(pipeline.unet, lora_config)

    # Training loop (simplified)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)
    unet.train()
    for epoch in range(epochs):
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            pixel_values = torch.cat([torch.unsqueeze(torch.tensor(tensor), 0) if not isinstance(tensor, torch.Tensor) else torch.unsqueeze(tensor, 0) for tensor in batch['pixel_values']]).to(device)
            captions = batch['generated_text']

            # Encode captions to get encoder_hidden_states
            text_inputs = pipeline.tokenizer(
                captions,
                padding="max_length",
                max_length=pipeline.tokenizer.model_max_length,  # Ensure it matches the model's max length
                truncation=True,  # Truncate captions that exceed the max length
                return_tensors="pt"
            )
            input_ids = text_inputs.input_ids.to(device)
            encoder_hidden_states = pipeline.text_encoder(input_ids)[0]

            # Convert pixel values to latent space
            with torch.no_grad():
                latents = pipeline.vae.encode(pixel_values).latent_dist.sample()  # Encode to latent space
                atents = latents * 0.18215  # Scale latent values (as required by Stable Diffusion)

            # Forward pass
            noise = torch.randn_like(latents)  # Generate noise in latent space
            timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device).long()  # Generate random timesteps
            noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)  # Add noise to latents
            optimizer.zero_grad()
            loss = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states)  # Pass timesteps
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}, Step {i+1}, Loss: {loss.item()}")

    # Save the LoRA weights
    unet.save_pretrained(output_dir)

if __name__ == "__main__":
    img_dir = "/dtu/blackhole/07/203495/ADLCV-project/data/ISIC_2019_Training_Input_10img"
    prompt_dir = "/dtu/blackhole/07/203495/ADLCV-project/data/ISIC_2019_Training_Input_10img_text/generated_descriptions.csv"
    train_lora(img_dir, prompt_dir, "lora_weights")
