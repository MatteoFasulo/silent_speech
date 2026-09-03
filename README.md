# Voicing Silent Speech

This repository contains code for synthesizing speech audio from silently mouthed words captured with electromyography (EMG).
This is a fork of the original [Gaddy's Silent Speech repository](https://github.com/dgaddy/silent_speech), updated to support the **TinyMyo** EMG foundation model for transduction and recognition tasks.

## Key Changes in this Fork

- **TinyMyo Integration**: Added support for pre-training and fine-tuning using the TinyMyo foundation model.
- **Improved Tooling**: Integrated [`uv`](https://docs.astral.sh/uv/) for faster dependency management and PyTorch setup.
- **Streamlined Dataset Pipeline**: Added HDF5 dataset building scripts for much faster training initialization.
- **WandB Logging**: Added experiment tracking via Weights & Biases.
- **Updated Decoding**: Replaced `DeepSpeech` dependencies with `SpeechBrain` and `torchaudio` built-in CTC decoders for better compatibility with modern hardware.

## Environment Setup

The code requires Python 3.10+. We recommend using `uv` to sync dependencies:

```bash
uv sync
```

You must also initialize submodules for Hifi-GAN and phoneme alignment data:

```bash
git submodule update --init
# Extract alignments if not already done
tar -xvzf text_alignments/text_alignments.tar.gz
```

## Data & Preprocessing

The EMG and audio data can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.4064408). We provide a script to automate this process via `download_data.py`. The audio must then be cleaned and resampled before building the HDF5 dataset; the preprocessing and training pipeline expects the generated resampled audio files.

An HDF5 dataset builder is also included to convert the raw data into a format that allows for much faster loading during training, suitable for HPC environments.

### 1. Download Data
Configure your `$DATA_PATH` in `config/transduction_model.json`, then run:
```bash
python download_data.py
```

### 2. Audio Cleaning and Resampling
Required before building the HDF5 dataset. This creates the resampled audio files used by preprocessing:
```bash
python data_collection/clean_audio.py
```

### 3. Build HDF5 Dataset
Required for performant training. This builds the dataset once rather than on-the-fly:
```bash
python build_hdf5.py
```

## EMG to Audio (Transduction)

This model synthesizes audio features (MFCCs) from EMG signals, which are then converted to wav via a vocoder.

### Training
```bash
python transduction_model.py
```
Configuration (hyperparameters, paths, WandB) is managed via `config/transduction_model.json`.

> **Note**: If the HiFi-GAN vocoder checkpoint is not found locally, the script will automatically download and extract it from Zenodo to ensure a seamless setup.

### Evaluation
To evaluate a saved model checkpoint and generate audio samples:
```bash
python transduction_model.py --evaluate_saved "./output/model_best.pt" --output_dir "./eval_results"
```

## EMG to Text (Recognition)

Directly convert silent speech to text using a CTC decoder.

For recognition, we use `SpeechBrain`'s built-in CTC decoder with a KenLM n-gram language model for improved performance. The `get_lexicon.py` script generates the necessary lexicon from the training data.

### Setup Decoder
Download the pre-trained KenLM language model and generate the custom lexicon from your dataset in one step:
```bash
python get_lexicon.py
```
This script downloads the Librispeech 4-gram model into the `KenLM/` directory and creates `gaddy_lexicon.txt` containing all unique words found in the local HDF5 datasets.

### Run
```bash
python recognition_model.py                                      # Train
python recognition_model.py --evaluate_saved "path/to/model.pt"  # Evaluate
```
Configuration (hyperparameters, paths, WandB) is managed via `config/recognition_model.json`.

## Documentation

This project uses `MkDocs` with the `Material` theme and `mkdocstrings` for API documentation.

To build and view the documentation locally:
```bash
# Serve the documentation
mkdocs serve
```
The documentation includes a Quick Start guide, detailed project sections, and automatically generated API references for all core modules.

## Resources
- **EMG Data**: [Zenodo (4064408)](https://doi.org/10.5281/zenodo.4064408)
- **Transduction Models**: [Zenodo (6747411)](https://doi.org/10.5281/zenodo.6747411)
- **Recognition Models**: [Zenodo (7183877)](https://doi.org/10.5281/zenodo.7183877)
