# build the lexicon from the dataset
import argparse
import logging
import os
import requests

import tqdm
from torchaudio.models.decoder import download_pretrained_files

from data_utils import TextTransform
from hdf5_dataset import H5EmgDataset

transform = TextTransform()


def get_unigram(dataset: H5EmgDataset) -> set:
    """
    Extracts all unique words (unigrams) from the dataset's text samples.

    Args:
        dataset: The H5EmgDataset to process.

    Returns:
        A set of unique lowercase words.
    """
    unigram = set()
    for example in tqdm.tqdm(dataset, "Building unigram", leave=False):
        clean = transform.clean_text(example["text"])
        for w in clean.split():
            unigram.add(w)
    return unigram


def get_lexicon(vocab: set, output_file: str) -> None:
    """
    Generates a lexicon file where each word is mapped to its character sequence.
    Format: word c h a r s |

    Args:
        vocab: Set of unique words.
        output_file: Path to save the lexicon.

    Returns:
        None. Writes the lexicon to the specified output file.
    """
    with open(output_file, "w", encoding="utf-8") as fout:
        for word in sorted(list(vocab)):
            # split word into char tokens
            chars = list(word)
            fout.write(f"{word} " + " ".join(chars) + " |\n")


def download_kenlm(output_dir: str) -> None:
    """
    Downloads pre-trained KenLM 4-gram language model files (lm.bin, lexicon.txt, tokens.txt).
    Uses torchaudio utility with a fallback to direct requests for systems with SSL issues.

    Args:
        output_dir: Directory where files will be saved.

    Returns:
        None. Files are saved to the specified output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    original_cwd = os.getcwd()
    os.chdir(output_dir)

    logging.info(f"Downloading KenLM files to {output_dir}...")
    try:
        download_pretrained_files("librispeech-4-gram")
    except Exception as e:
        logging.warning(f"Torchaudio download failed ({e}), attempting manual download...")
        files = {
            "lm.bin": "https://download.pytorch.org/torchaudio/decoder-assets/librispeech-4-gram/lm.bin",
            "lexicon.txt": "https://download.pytorch.org/torchaudio/decoder-assets/librispeech-4-gram/lexicon.txt",
            "tokens.txt": "https://download.pytorch.org/torchaudio/decoder-assets/librispeech-4-gram/tokens.txt",
        }
        for filename, url in files.items():
            if not os.path.exists(filename):
                logging.info(f"Downloading {filename}...")
                response = requests.get(url, stream=True, timeout=60)
                response.raise_for_status()
                with open(filename, "wb") as f:
                    f.write(response.content)

    os.chdir(original_cwd)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Setup KenLM decoder: download LM and build custom lexicon.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./KenLM",
        help="Directory to save KenLM files (default: KenLM)",
    )
    args = parser.parse_args()

    # 1. Download pre-trained files
    download_kenlm(args.output_dir)

    # 2. Build custom lexicon from dataset
    logging.info("Scanning dataset for unique words to build custom lexicon...")
    trainset = H5EmgDataset(dev=False, test=False)
    devset = H5EmgDataset(dev=True, test=False)
    testset = H5EmgDataset(dev=False, test=True)

    merged_unigram = get_unigram(trainset) | get_unigram(devset) | get_unigram(testset)

    lexicon_path = os.path.join(args.output_dir, "gaddy_lexicon.txt")
    get_lexicon(merged_unigram, output_file=lexicon_path)

    logging.info(f"Setup complete!")
    logging.info(f"  - Pre-trained LM and tokens are in {args.output_dir}")
    logging.info(f"  - Custom lexicon saved to {lexicon_path} ({len(merged_unigram)} words)")
