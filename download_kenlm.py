"""Download the KenLM language model used by the decoder."""

import argparse
import logging
import os
import shutil

import requests
from torchaudio.models.decoder import download_pretrained_files


def download_kenlm_model(output_dir: str) -> str:
    """Download lm.bin into output_dir and return its path."""
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "lm.bin")

    logging.info("Downloading KenLM model to %s", output_path)

    try:
        downloaded = download_pretrained_files("librispeech-4-gram")
        shutil.copy2(downloaded.lm, output_path)
    except Exception as error:
        logging.warning("Torchaudio download failed (%s); using direct download.", error)
        url = "https://download.pytorch.org/torchaudio/decoder-assets/librispeech-4-gram/lm.bin"
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with open(output_path, "wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)

    logging.info("KenLM model saved to %s", output_path)
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Download the KenLM language model.")
    parser.add_argument(
        "--output_dir",
        default="./KenLM",
        help="Directory in which to save lm.bin.",
    )
    args = parser.parse_args()
    download_kenlm_model(args.output_dir)

