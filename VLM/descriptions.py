import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image, ImageOps
from transformers import AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained("AIDC-AI/Ovis2-4B",
                                             torch_dtype=torch.bfloat16,
                                             multimodal_max_length=32768,
                                             trust_remote_code=True).cuda()
text_tokenizer = model.get_text_tokenizer()
visual_tokenizer = model.get_visual_tokenizer()

# Load metadata and ground truth
#for full dataset
metadata_df = pd.read_csv('/work3/lucor/ISIC_2019_Training_Metadata.csv')
ground_truth_df = pd.read_csv('/work3/lucor/ISIC_2019_Training_GroundTruth.csv')

#demo 10
# metadata_df = pd.read_csv('/zhome/ec/c/204596/ADLCV-project/data/ISIC_2019_Training_Metadata.csv')
# ground_truth_df = pd.read_csv('/zhome/ec/c/204596/ADLCV-project/data/ISIC_2019_Training_GroundTruth.csv')

# Convert to dictionaries for fast lookup
metadata_dict = metadata_df.set_index("image").to_dict(orient="index")
ground_truth_dict = ground_truth_df.set_index("image").to_dict(orient="index")

# Path to folder containing images for full dataset
image_folder = '/work3/lucor/ISIC_2019_Training_Input'

# Path to folder containing images for trial 10
# image_folder = '/zhome/ec/c/204596/ADLCV-project/data/ISIC_2019_Training_Input'

#output of csv
output_csv = '/work3/lucor/test_moreexamples.csv'

# Store results in a list
results = []

#Limit the number of images to process
# MAX_IMAGES = 20
# processed = 0

# Iterate over images in the folder
for filename in sorted(os.listdir(image_folder)):
    if not filename.endswith(".jpg"):
        continue  # Skip non-image files

    # if processed >= MAX_IMAGES:
    #     break  # Stop after processing MAX_IMAGES

    image_id = os.path.splitext(filename)[0]  # Extract ID without extension
    image_path = os.path.join(image_folder, filename)

    # Load the image
    image = Image.open(image_path)

    # Resize to fit within 448x448, maintaining the aspect ratio (no padding)
    image = ImageOps.contain(image, (448, 448))

    images = [image]  # List to pass to the model

    # Retrieve metadata and ground truth    
    if image_id in ground_truth_dict:
        disease_scores = ground_truth_dict[image_id]
        
        # Check if all values are zero
        if all(value == 0 for value in disease_scores.values()):
            disease = "Unknown"
        else:
            disease = max(disease_scores, key=disease_scores.get)

    meta = metadata_dict.get(image_id, {})
    age = str(meta.get("age_approx", "Unknown"))
    sex = str(meta.get("sex", "Unknown")).lower()
    location = str(meta.get("anatom_site_general", "Unknown")).lower()

    # query = (
    #     f"<image>\n"
    #     f"Describe in english, and in maximum two sentences, the medical image of a {sex} patient, approximately {age} years old, "
    #     f"with a lesion located on the {location}. The diagnosed condition is {disease}."
    # )

    query = (
        f"<image>\n"
        f"Convert {sex}'s {age}y/o {location} lesion ({disease}) into concise JSON values. "
        f"Order: [age, gender, location, disease, visual_description]. "
        f"Description: 25-30 word factual observation. No explanations, adverbs, or hedging. "
        f"Format: [num, str, str, str, str]. No keys. "
        f"Examples:"
        f"[55.0,female,anterior torso,NV,'irregular dark brown lesion with variegated pigmentation, slightly raised surface, and fuzzy borders, surrounded by pale halo'], "
        f"[34, female, leg, NV, 'Symmetrical oval lesion with uniform light brown pigmentation, regular network pattern at periphery, faint central hypopigmentation, 5mm diameter'], "
        f"[71, male, face, BCC, 'pearly pink plaque with arborizing telangiectasia, rolled borders, central ulceration covered by crust, 15mm largest dimension'],"
        f"[58, male, back, MEL, 'highly irregular lesion with jagged borders showing uneven distribution of dark brown, black, and reddish tones, with scattered blue-gray areas and subtle white scar-like patches'],"
        f"[32, female, arm, NV, 'uniform tan-brown oval lesion with delicate pigment network radiating from center, smooth surface texture, and gradual fading at edges'],"
        f"[71, male, nose, BCC, 'shiny pinkish-white nodule with fine branching blood vessels, translucent pearly border, and small central crust surrounded by rolled edges'],"
        f"[65, female, cheek, AK, 'rough-textured pinkish patch with gritty yellow-white scale that feels like sandpaper, on sun-damaged skin showing surrounding telangiectasia'],"
        f"[60, male, chest, BKL, 'slightly elevated waxy brown plaque with cracked surface resembling dried paint, showing stuck-on appearance with sharp demarcation from surrounding skin'],"
        f"[45, female, leg, DF, 'firm dome-shaped nodule with characteristic dimpling when pinched, displaying tan-brown periphery and darker central zone with subtle scale'],"
        f"[50, male, lip, VASC, 'vivid red rubbery papule with smooth glassy surface and radiating blood vessels at periphery, blanching partially under pressure'],"
        f"[68, female, ear, SCC, 'thickened scaly growth with uneven crusted surface showing areas of yellowish keratin and focal bright red erosions']"
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
            max_new_tokens=70,
            do_sample=False,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=text_tokenizer.pad_token_id,
            use_cache=True
        )
        output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
        output = text_tokenizer.decode(output_ids, skip_special_tokens=True)

    # Append results
    results.append([image_id, output])
    # processed += 1  # Increment counter

# Save results to CSV
results_df = pd.DataFrame(results, columns=["image_id", "generated_text"])
results_df.to_csv(output_csv, index=False, sep='\t')

print(f"Generated descriptions saved to {output_csv}")