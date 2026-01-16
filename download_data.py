import os
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

from data_utils import load_config

CONFIG = load_config(os.path.join("config", "transduction_model.json"))

if __name__ == "__main__":
    output_dir = Path(os.path.expandvars(CONFIG.data_root)).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = "emg_data.tar.gz"

    # Check if the dataset is already downloaded
    if not os.path.exists(os.path.join(output_dir, filename)):
        # Download the dataset
        print(f"Downloading Silent Speech dataset to {output_dir}...")
        response = requests.get(f"https://zenodo.org/records/4064409/files/{filename}", stream=True, timeout=60)
        # Sizes in bytes.
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024
        with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
            with open(os.path.join(output_dir, filename), "wb") as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
        print("Download completed.")

    # Extract the dataset
    print("Extracting the dataset...")
    if tarfile.is_tarfile(os.path.join(output_dir, filename)):
        with tarfile.open(os.path.join(output_dir, filename), "r:gz") as tar:
            for member in tqdm(tar.getmembers(), total=len(tar.getmembers()), desc="Extracting files"):
                tar.extract(member=member, path=output_dir)
    print("Extraction completed.")
