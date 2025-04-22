from diffusers import StableDiffusionPipeline
from peft import PeftModel, PeftConfig
import torch
import csv
import os
import pandas as pd
from tqdm import tqdm  # Import tqdm for the progress bar

def generate_images_from_csv(csv_path, model_path, lora_weights, output_dir="image_generation/results_isic_5000_768", output_csv="image_generation/results_isic_5000_768/generated_images.csv"):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the Stable Diffusion pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None  # Disable the NSFW filter
    ).to("cuda")

    # Load LoRA weights into the UNet
    peft_config = PeftConfig.from_pretrained(lora_weights)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_weights, adapter_name="default")

    # Load the CSV file
    df = pd.read_csv(csv_path, delimiter='\t')
    df.columns = df.columns.str.strip('"')  # Remove quotes from column names if present

    # Ensure the required columns exist
    if 'image_id' not in df.columns or 'generated_text' not in df.columns:
        raise KeyError("The CSV file must contain 'image_id' and 'generated_text' columns.")

    # Open the output CSV file for logging
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, mode="a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(["Image File Name", "Prompt"])  # Write header if file doesn't exist

        # Iterate through each row in the CSV with a progress bar
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Generating Images", unit="image"):
            prompt = row['generated_text']
            image_id = row['image_id']

            # Generate the image
            image = pipe(prompt, height=768, width=768).images[0]

            # Save the image with a unique name in the results folder
            image_file_name = f"{image_id}_generated.png"
            image_path = os.path.join(output_dir, image_file_name)
            image.save(image_path)

            # Log the image file name and prompt in the CSV
            writer.writerow([image_file_name, prompt])

if __name__ == "__main__":
    # Define paths
    csv_path = "/dtu/blackhole/07/203495/ADLCV-project/data/ISIC_2019_Training_Input/generated_descriptions_isic_5000.csv"
    model_path = "runwayml/stable-diffusion-v1-5"  # Base model path
    lora_weights = "/dtu/blackhole/07/203495/ADLCV-project/image_generation/lora_weights_isic_5000"  # Path to LoRA weights

    # Generate images for all prompts in the CSV
    generate_images_from_csv(csv_path, model_path, lora_weights)