import argparse
import os
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Silent Speech dataset from Zenodo."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the downloaded dataset.",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = "emg_data.tar.gz"

    # Download the dataset
    print(f"Downloading Silent Speech dataset to {output_dir}...")
    os.system(
        f"wget -c https://zenodo.org/records/4064409/files/{filename} -O {os.path.join(output_dir, filename)}"
    )
    print("Download completed.")

    # Extract the dataset
    os.system(f"tar -xvzf {os.path.join(output_dir, filename)} -C {output_dir}")
    print("Extraction completed.")
