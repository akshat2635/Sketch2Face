import requests
import os
import zipfile
from tqdm import tqdm

def download_file(url, save_path):
    """Downloads a file only if it does not already exist."""
    if not os.path.exists(save_path):
        print(f"Downloading {os.path.basename(save_path)}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(save_path, "wb") as f, tqdm(
            desc=os.path.basename(save_path), total=total_size, unit='B', unit_scale=True
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        return 1
    else:
        print(f"{os.path.basename(save_path)} already exists. Skipping download.")
        return 0

def extract_zip(zip_path, extract_to):
    """Extracts a ZIP file only if the extracted directory does not already exist."""
    extracted_dir = os.path.join(extract_to, os.path.splitext(os.path.basename(zip_path))[0])
    if not os.path.exists(extracted_dir):
        print(f"Extracting {os.path.basename(zip_path)}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
    else:
        print(f"{os.path.basename(zip_path)} already extracted. Skipping extraction.")

# Dataset URLs
image_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"

# Create dataset directory
os.makedirs("flickr8k", exist_ok=True)

# Define file paths
image_zip_path = "flickr8k/Flickr8k_Dataset.zip"

# Download and extract images
res=download_file(image_url, image_zip_path)
if res:
    extract_zip(image_zip_path, "flickr8k")