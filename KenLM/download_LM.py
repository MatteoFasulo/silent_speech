import os

import requests
from torchaudio.models.decoder import download_pretrained_files

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download KenLM language model files for ASR decoding")
    parser.add_argument(
        "--output_directory",
        type=str,
        required=True,
        help="Directory to save the downloaded language model files",
    )
    args = parser.parse_args()
    os.makedirs(args.output_directory, exist_ok=True)
    os.chdir(args.output_directory)

    try:
        download_pretrained_files("librispeech-4-gram")

    except Exception:  # SSL certificate errors on some systems
        response = requests.get(
            "https://download.pytorch.org/torchaudio/decoder-assets/librispeech-4-gram/lm.bin", stream=True, timeout=60
        )
        with open("lm.bin", "wb") as file:
            file.write(response.content)
        response = requests.get(
            "https://download.pytorch.org/torchaudio/decoder-assets/librispeech-4-gram/lexicon.txt",
            stream=True,
            timeout=60,
        )
        with open("lexicon.txt", "wb") as file:
            file.write(response.content)
        response = requests.get(
            "https://download.pytorch.org/torchaudio/decoder-assets/librispeech-4-gram/tokens.txt",
            stream=True,
            timeout=60,
        )
        with open("tokens.txt", "wb") as file:
            file.write(response.content)

    print("Language model files downloaded.")
