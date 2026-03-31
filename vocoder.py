import json
import os
import shutil
import sys
import zipfile

import requests
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from data_utils import load_config

sys.path.append("./hifi_gan")
from env import AttrDict

from models import Generator

FLAGS = load_config(os.path.join("config", "transduction_model.json"))


def download_and_extract_pretrained(url: str, dest_dir: str) -> None:
    """
    Downloads a zip file from a URL, extracts it, and moves the HiFi-GAN model
    to the expected location.

    Args:
        url (str): The URL to download the zip file from.
        dest_dir (str): The destination directory (e.g., './').
    """
    temp_zip = "pretrained_models.zip"
    extract_temp = "temp_pretrained_extract"

    print(f"Downloading pre-trained models from {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with tqdm(total=total_size, unit="B", unit_scale=True, desc=temp_zip) as progress_bar:
        with open(temp_zip, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

    print("Extracting models...")
    with zipfile.ZipFile(temp_zip, "r") as zip_ref:
        zip_ref.extractall(extract_temp)

    # The zip contains 'pretrained_models/hifigan_finetuned'
    source_path = os.path.join(extract_temp, "pretrained_models", "hifigan_finetuned")
    target_path = os.path.join(dest_dir, "hifigan_finetuned")

    if os.path.exists(source_path):
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
        shutil.move(source_path, target_path)
        print(f"Moved HiFi-GAN model to {target_path}")

    # Cleanup
    os.remove(temp_zip)
    shutil.rmtree(extract_temp)
    print("Cleanup completed.")


class Vocoder(object):
    def __init__(self, device: str = "cuda"):
        checkpoint_file = FLAGS.hifigan_checkpoint
        if checkpoint_file is None:
            raise ValueError("hifigan_checkpoint must be specified in the configuration.")

        # URLs for pre-trained HiFi-GAN from Zenodo (6747411)
        zip_url = "https://zenodo.org/records/6747411/files/pretrained_models.zip?download=1"
        config_file = os.path.join(os.path.dirname(checkpoint_file), "config.json")

        if not os.path.exists(checkpoint_file) or not os.path.exists(config_file):
            print("HiFi-GAN models not found. Downloading...")
            # We assume hifigan_finetuned should be in the directory containing checkpoint_file's parent or similar
            # Based on the zip structure, we extract to the current working directory
            download_and_extract_pretrained(zip_url, ".")

        with open(config_file) as f:
            hparams = AttrDict(json.load(f))
        self.generator = Generator(hparams).to(device)
        state_dict = torch.load(checkpoint_file, map_location=device)["generator"]
        self.generator.load_state_dict(state_dict)
        self.generator.eval()
        self.generator.remove_weight_norm()

    def __call__(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Generates audio from a mel-spectrogram.

        Args:
            mel_spectrogram (torch.Tensor): Mel-spectrogram tensor of shape (seq_len, 80).

        Returns:
            torch.Tensor: 1D audio tensor.
        """
        with torch.no_grad():
            mel_spectrogram = mel_spectrogram.T[np.newaxis, :, :]
            audio = self.generator(mel_spectrogram)
        return audio.squeeze()
