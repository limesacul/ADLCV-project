import os
import numpy as np
import csv
from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
import torch

def load_image(image_path):
    """Load an image and convert it to a numpy array."""
    try:
        image = Image.open(image_path).convert("RGB")
        return np.array(image)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def calculate_fid_kid(folder1, folder2, device="cuda"):
    """Calculate FID and KID metrics between two folders."""
    fid = FrechetInceptionDistance(feature=2048).to(device)

    # Initialize KID with a default subset_size of 50
    kid = KernelInceptionDistance(subset_size=50).to(device)

    valid_pairs = 0  # Counter for valid image pairs

    print(f"Starting FID/KID calculation...")
    print(f"Original images folder: {folder1}")
    print(f"Generated images folder: {folder2}")

    for filename in tqdm(os.listdir(folder2), desc="Calculating FID/KID"):
        if filename.endswith("_generated.png"):
            print(f"Processing generated file: {filename}")
            original_filename = filename.replace("_generated.png", ".jpg")
            base_image_path = os.path.join(folder1, original_filename)
            generated_image_path = os.path.join(folder2, filename)

            print(f"Looking for original file: {original_filename}")
            if not os.path.exists(base_image_path):
                print(f"Original file not found: {base_image_path}, skipping...")
                continue

            base_image = load_image(base_image_path)
            generated_image = load_image(generated_image_path)

            if base_image is None:
                print(f"Failed to load original image: {base_image_path}")
                continue
            if generated_image is None:
                print(f"Failed to load generated image: {generated_image_path}")
                continue

            # Convert images to tensors with dtype=torch.uint8 and move to the correct device
            base_image_tensor = torch.tensor(base_image).permute(2, 0, 1).unsqueeze(0).to(torch.uint8).to(device)
            generated_image_tensor = torch.tensor(generated_image).permute(2, 0, 1).unsqueeze(0).to(torch.uint8).to(device)

            # Update FID and KID
            fid.update(base_image_tensor, real=True)
            fid.update(generated_image_tensor, real=False)
            kid.update(base_image_tensor, real=True)
            kid.update(generated_image_tensor, real=False)

            valid_pairs += 1
            print(f"Valid pair found: {original_filename} and {filename}")

    print(f"Total valid pairs found: {valid_pairs}")

    if valid_pairs < 1:
        raise RuntimeError("No valid image pairs found to compute FID and KID.")

    # Dynamically adjust subset_size for KID if necessary
    if valid_pairs < 50:
        print(f"Adjusting KID subset_size to {valid_pairs} (number of valid pairs)")
        kid = KernelInceptionDistance(subset_size=valid_pairs).to(device)

    fid_score = fid.compute().item()
    kid_mean, kid_std = kid.compute()
    print(f"FID Score: {fid_score}")
    print(f"KID Mean: {kid_mean}, KID Std: {kid_std}")
    return fid_score, kid_mean.item(), kid_std.item()

def calculate_ssim_psnr(folder1, folder2):
    """Calculate SSIM and PSNR metrics between two folders."""
    ssim_scores = []
    psnr_scores = []

    print(f"Starting SSIM/PSNR calculation...")
    print(f"Original images folder: {folder1}")
    print(f"Generated images folder: {folder2}")

    for filename in tqdm(os.listdir(folder2), desc="Calculating SSIM/PSNR"):
        if filename.endswith("_generated.png"):
            print(f"Processing generated file: {filename}")
            original_filename = filename.replace("_generated.png", ".jpg")
            base_image_path = os.path.join(folder1, original_filename)
            generated_image_path = os.path.join(folder2, filename)

            print(f"Looking for original file: {original_filename}")
            if not os.path.exists(base_image_path):
                print(f"Original file not found: {base_image_path}, skipping...")
                continue

            base_image = load_image(base_image_path)
            generated_image = load_image(generated_image_path)

            if base_image is None:
                print(f"Failed to load original image: {base_image_path}")
                continue
            if generated_image is None:
                print(f"Failed to load generated image: {generated_image_path}")
                continue

            # Resize the generated image to match the dimensions of the original image
            if base_image.shape != generated_image.shape:
                print(f"Resizing generated image from {generated_image.shape} to {base_image.shape}")
                generated_image = np.array(Image.fromarray(generated_image).resize((base_image.shape[1], base_image.shape[0])))

            # Calculate SSIM and PSNR
            ssim_score = ssim(base_image, generated_image, multichannel=True)
            psnr_score = psnr(base_image, generated_image)

            ssim_scores.append(ssim_score)
            psnr_scores.append(psnr_score)
            print(f"SSIM: {ssim_score}, PSNR: {psnr_score}")

    print(f"Total SSIM scores: {len(ssim_scores)}, Total PSNR scores: {len(psnr_scores)}")
    return np.mean(ssim_scores), np.mean(psnr_scores)

def compare_images(folder1, folder2):
    """Compare images in two folders using FID, KID, SSIM, and PSNR."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Calculating FID and KID...")
    fid_score, kid_mean, kid_std = calculate_fid_kid(folder1, folder2, device)

    # print("Calculating SSIM and PSNR...")
    # ssim_score, psnr_score = calculate_ssim_psnr(folder1, folder2)

    print("\nComparison Results:")
    print(f"FID: {fid_score:.4f}")
    print(f"KID: {kid_mean:.4f} ± {kid_std:.4f}")
    # print(f"SSIM: {ssim_score:.4f}")
    # print(f"PSNR: {psnr_score:.4f} dB")

    # Save results to a CSV file in the generated images folder
    results_csv_path = os.path.join(folder2, "comparison_results.csv")
    with open(results_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["FID", f"{fid_score:.4f}"])
        writer.writerow(["KID Mean", f"{kid_mean:.4f}"])
        writer.writerow(["KID Std", f"{kid_std:.4f}"])
        # writer.writerow(["SSIM", f"{ssim_score:.4f}"])
        # writer.writerow(["PSNR", f"{psnr_score:.4f} dB"])

    print(f"\nResults saved to {results_csv_path}")

if __name__ == "__main__":
    # Define the paths to the two folders
    folder1 = "/dtu/blackhole/07/203495/ADLCV-project/data/ISIC_2019_Training_Input"
    folder2 = "/dtu/blackhole/07/203495/ADLCV-project/image_generation/results_isic_5000_test5"

    # Compare the images
    compare_images(folder1, folder2)