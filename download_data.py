import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import requests
from tqdm import tqdm

from data_utils import load_config

CONFIG = load_config(os.path.join("config", "transduction_model.json"))

# Resolve Zenodo's UUID-backed file URL from the record API. This avoids the
# record UI file route, which may return 403 to scripted clients.
ZENODO_RECORD_URL = "https://zenodo.org/api/records/4064409"
DATASET_FILENAME = "emg_data.tar.gz"
DATASET_MD5 = "7f97d2182b896652999b1b2d0c69fd7b"


def md5sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_download_url() -> str:
    """Resolve Zenodo's UUID-backed file URL from the record metadata."""
    headers = {"User-Agent": "silent-speech-dataset-downloader/1.0"}
    with requests.get(ZENODO_RECORD_URL, headers=headers, timeout=(30, 60)) as response:
        response.raise_for_status()
        record = response.json()

    for file_entry in record.get("files", []):
        if file_entry.get("key") != DATASET_FILENAME:
            continue
        file_url = file_entry.get("links", {}).get("self")
        if file_url:
            return file_url

    raise RuntimeError(f"Could not find {DATASET_FILENAME!r} in Zenodo record metadata.")


def download_dataset(archive_path: Path) -> None:
    partial_path = archive_path.with_name(f"{archive_path.name}.part")
    print(f"Downloading Silent Speech dataset to {archive_path.parent}...")

    try:
        dataset_url = resolve_download_url()
        print(f"Resolved Zenodo download URL: {dataset_url}")
        headers = {
            "User-Agent": "silent-speech-dataset-downloader/1.0",
        }
        with requests.get(dataset_url, headers=headers, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))

            with tqdm(
                total=total_size or None,
                unit="B",
                unit_scale=True,
                desc="Downloading",
            ) as progress_bar:
                with partial_path.open("wb") as file:
                    for data in response.iter_content(chunk_size=1024 * 1024):
                        if data:
                            file.write(data)
                            progress_bar.update(len(data))

        actual_md5 = md5sum(partial_path)
        if actual_md5 != DATASET_MD5:
            raise RuntimeError(
                f"Downloaded archive checksum mismatch: expected {DATASET_MD5}, got {actual_md5}."
            )

        # Replace the final path only after the complete archive has been verified.
        partial_path.replace(archive_path)
        print("Download completed and verified.")
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    configured_data_root = os.path.expandvars(CONFIG.data_root)
    if "$" in configured_data_root:
        raise RuntimeError(
            f"DATA_PATH is not set or contains an unresolved variable: {CONFIG.data_root}"
        )

    output_dir = Path(configured_data_root).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_path = output_dir / DATASET_FILENAME

    # Reuse an existing archive only if it is complete and verified.
    if archive_path.exists() and md5sum(archive_path) == DATASET_MD5:
        print(f"Using verified archive: {archive_path}")
    else:
        download_dataset(archive_path)

    # Extract with native GNU tar. This avoids Python-level work for every file;
    # pigz parallelizes gzip decompression when it is installed.
    print("Extracting the dataset...")

    tar_command = ["tar"]
    if shutil.which("pigz"):
        tar_command.extend(["--use-compress-program=pigz"])
    # Verbose mode prints each extracted path, providing exact live status.
    tar_command.extend(
        [
            "-xvf",
            str(archive_path),
            "-C",
            str(output_dir),
        ]
    )

    try:
        subprocess.run(tar_command, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Could not extract archive: {archive_path}") from error

    print("Extraction completed.")
