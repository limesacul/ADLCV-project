import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from transformers import AutoModelForCausalLM

# load model
model = AutoModelForCausalLM.from_pretrained("AIDC-AI/Ovis2-1B",
                                             torch_dtype=torch.bfloat16,
                                             multimodal_max_length=32768,
                                             trust_remote_code=True).cuda()
text_tokenizer = model.get_text_tokenizer()
visual_tokenizer = model.get_visual_tokenizer()

# Load metadata and ground truth separately
metadata_df = pd.read_csv('/zhome/ec/c/204596/ADLCV-project/data/ISIC_2019_Training_Metadata.csv')
ground_truth_df = pd.read_csv('/zhome/ec/c/204596/ADLCV-project/data/ISIC_2019_Training_GroundTruth.csv')

# Convert to dictionaries for fast lookup
metadata_dict = metadata_df.set_index("image").to_dict(orient="index")
ground_truth_dict = ground_truth_df.set_index("image").to_dict(orient="index")


class MedicalImageDataset(Dataset):
    def __init__(self, img_dir, metadata_dict, ground_truth_dict, text_tokenizer, visual_tokenizer):
        self.img_dir = img_dir
        self.metadata_dict = metadata_dict
        self.ground_truth_dict = ground_truth_dict
        self.text_tokenizer = text_tokenizer
        self.visual_tokenizer = visual_tokenizer
        self.image_ids = list(ground_truth_dict.keys())  # Image list from ground truth

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = f"{self.img_dir}/{image_id}.jpg"  # Assuming images are named as 'ISIC_0000000.jpg'
        image = Image.open(image_path)

        # Get disease label
        disease = max(self.ground_truth_dict[image_id], key=self.ground_truth_dict[image_id].get)  # Highest probability disease

        # Get patient metadata
        meta = self.metadata_dict.get(image_id, {})  # Default to empty if missing
        age = meta.get("age_approx", "Unknown")
        sex = meta.get("sex", "Unknown")
        location = meta.get("anatom_site_general", "Unknown")

        # Format prompt
        prompt = f"<image>\nPatient Info: Age {age}, {sex}, Location {location}\nDiagnosed Condition: {disease}\nDescribe the image."

        # Tokenize inputs
        prompt, input_ids, pixel_values = model.preprocess_inputs(prompt, [image], max_partition=9)
        attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)

        return {
            "input_ids": input_ids.unsqueeze(0),
            "pixel_values": pixel_values.to(dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device),
            "attention_mask": attention_mask.unsqueeze(0)
        }


# Initialize dataset and dataloader
img_dir = "/zhome/ec/c/204596/ADLCV-project/data/ISIC_2019_Training_Input"
dataset = MedicalImageDataset(img_dir, metadata_dict, ground_truth_dict, text_tokenizer, visual_tokenizer)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Sample inference
for batch in dataloader:
    with torch.inference_mode():
        output_ids = model.generate(
            batch["input_ids"].to(model.device),
            pixel_values=[batch["pixel_values"]],
            attention_mask=batch["attention_mask"].to(model.device),
            max_new_tokens=1024,
            do_sample=False
        )[0]
        output = text_tokenizer.decode(output_ids, skip_special_tokens=True)
        print(f'Output:\n{output}')
    break  # Process only one sample for testing