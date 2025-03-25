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

# Get disease label
disease = max(ground_truth_dict['ISIC_0000000'], key=ground_truth_dict['ISIC_0000000'].get)  # Highest probability disease

# Get patient metadata
meta = metadata_dict.get('ISIC_0000000', {})  # Default to empty if missing
age = meta.get("age_approx", "Unknown")
sex = meta.get("sex", "Unknown")
location = meta.get("anatom_site_general", "Unknown")

# single-image input
image_path = '/zhome/ec/c/204596/ADLCV-project/data/ISIC_2019_Training_Input/ISIC_0000000.jpg'
images = [Image.open(image_path)]
max_partition = 9
text = 'Describe the image.'
query = f"<image>\nPatient Info: Age {age}, {sex}, Location {location}\nDiagnosed Condition: {disease}\nDescribe the image."


# format conversation
prompt, input_ids, pixel_values = model.preprocess_inputs(query, images, max_partition=max_partition)
attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
input_ids = input_ids.unsqueeze(0).to(device=model.device)
attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
if pixel_values is not None:
    pixel_values = pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)
pixel_values = [pixel_values]

# generate output
with torch.inference_mode():
    gen_kwargs = dict(
        max_new_tokens=1024,
        do_sample=False,
        top_p=None,
        top_k=None,
        temperature=None,
        repetition_penalty=None,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=text_tokenizer.pad_token_id,
        use_cache=True
    )
    output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
    output = text_tokenizer.decode(output_ids, skip_special_tokens=True)
    print(f'Output:\n{output}')