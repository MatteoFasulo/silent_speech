import os

from torchaudio.models.decoder import download_pretrained_files

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download KenLM language model files for ASR decoding"
    )
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

    except Exception as e:  # SSL certificate errors on some systems
        os.system(
            "wget https://download.pytorch.org/torchaudio/decoder-assets/librispeech-4-gram/lm.bin"
        )
        os.system(
            "wget https://download.pytorch.org/torchaudio/decoder-assets/librispeech-4-gram/lexicon.txt"
        )
        os.system(
            "wget https://download.pytorch.org/torchaudio/decoder-assets/librispeech-4-gram/tokens.txt"
        )

    print("Language model files downloaded.")
