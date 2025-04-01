import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
from transformers import AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained("AIDC-AI/Ovis2-1B",
                                             torch_dtype=torch.bfloat16,
                                             multimodal_max_length=32768,
                                             trust_remote_code=True).cuda()
text_tokenizer = model.get_text_tokenizer()
visual_tokenizer = model.get_visual_tokenizer()

# Load metadata and ground truth
metadata_df = pd.read_csv('/zhome/ec/c/204596/ADLCV-project/data/ISIC_2019_Training_Metadata.csv')
ground_truth_df = pd.read_csv('/zhome/ec/c/204596/ADLCV-project/data/ISIC_2019_Training_GroundTruth.csv')

# Convert to dictionaries for fast lookup
metadata_dict = metadata_df.set_index("image").to_dict(orient="index")
ground_truth_dict = ground_truth_df.set_index("image").to_dict(orient="index")

# Path to folder containing images
image_folder = '/zhome/ec/c/204596/ADLCV-project/data/ISIC_2019_Training_Input'

# Iterate over images in the folder
for filename in sorted(os.listdir(image_folder)):
    if not filename.endswith(".jpg"):
        continue  # Skip non-image files

    image_id = os.path.splitext(filename)[0]  # Extract ID without extension
    image_path = os.path.join(image_folder, filename)
    
    # Load the image
    images = [Image.open(image_path)]

    # Retrieve metadata and ground truth    
    if image_id in ground_truth_dict:
        disease_scores = ground_truth_dict[image_id]
        
        # Check if all values are zero
        if all(value == 0 for value in disease_scores.values()):
            disease = "Unknown"
        else:
            disease = max(disease_scores, key=disease_scores.get)

    meta = metadata_dict.get(image_id, {})
    age = meta.get("age_approx", "Unknown")
    sex = meta.get("sex", "Unknown")
    location = meta.get("anatom_site_general", "Unknown")

    # Construct query
    query = (
        f"<image>\n"
        f"Describe the medical image of a {sex.lower()} patient, approximately {age} years old, "
        f"with a lesion located on the {location.lower()}. The diagnosed condition is {disease}."
    )   

    # Format conversation
    prompt, input_ids, pixel_values = model.preprocess_inputs(query, images, max_partition=9)
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    input_ids = input_ids.unsqueeze(0).to(device=model.device)
    attention_mask = attention_mask.unsqueeze(0).to(device=model.device)

    if pixel_values is not None:
        pixel_values = pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)
    pixel_values = [pixel_values]

    # Generate output
    with torch.inference_mode():
        gen_kwargs = dict(
            max_new_tokens=512,
            do_sample=False,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=text_tokenizer.pad_token_id,
            use_cache=True
            temperature=0.8       # Adds slight randomness
        )
        output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
        output = text_tokenizer.decode(output_ids, skip_special_tokens=True)

    print(f'Image: {filename}\nOutput:\n{output}\n{"-"*50}')