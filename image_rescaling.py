import os
from PIL import Image
from tqdm import tqdm

def rescale_images(input_folder, new_size=(128, 128)):
    # Create the output folder in the same directory as the input folder
    output_folder = f"{input_folder}_{new_size[0]}x{new_size[1]}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get the list of files to process
    files = [f for f in os.listdir(input_folder) if f.endswith(".jpg")]

    # Iterate through all files with a progress bar
    for filename in tqdm(files, desc="Rescaling images"):
        input_path = os.path.join(input_folder, filename)
        output_filename = f"{os.path.splitext(filename)[0]}_{new_size[0]}x{new_size[1]}.jpg"
        output_path = os.path.join(output_folder, output_filename)

        # Open the image, rescale it, and save it to the output folder
        with Image.open(input_path) as img:
            img_rescaled = img.resize(new_size)
            img_rescaled.save(output_path)

if __name__ == "__main__":
    input_folder = "/dtu/blackhole/07/203495/ADLCV-project/data/ISIC_2019_Training_Input"
    rescale_images(input_folder)
