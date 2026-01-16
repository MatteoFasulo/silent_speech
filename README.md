# Voicing Silent Speech

This repository contains code for synthesizing speech audio from silently mouthed words captured with electromyography (EMG).
It is the official repository for the papers [Digital Voicing of Silent Speech](https://aclanthology.org/2020.emnlp-main.445.pdf) at EMNLP 2020, [An Improved Model for Voicing Silent Speech](https://aclanthology.org/2021.acl-short.23.pdf) at ACL 2021, and the dissertation [Voicing Silent Speech](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2022/EECS-2022-68.pdf).
The current commit contains only the most recent model, but the versions from prior papers can be found in the commit history.
On an ASR-based open vocabulary evaluation, the latest model achieves a WER of approximately 36%.
Audio samples can be found [here](https://dgaddy.github.io/silent_speech_samples/June2022/).

The repository also includes code for directly converting silent speech to text.  See the section labeled [Silent Speech Recognition](#silent-speech-recognition).

## Changes from Prior Versions

Compared to the original code several changes have been made to improve usability of the codebase:

- We added a script to download the data directly into the expected directory structure.
- The environment setup instructions have been updated to use [`uv`](https://docs.astral.sh/uv/) to simplify installation of PyTorch with CUDA support. You can still use yout preferred method to install dependencies if desired via the `requirements.txt` or `pyproject.toml` files.
- The audio cleaning step has been improved by running the cleaning in parallel and saving cleaned audio files to avoid re-sampling on each training run.
- Instead of building the dataset on-the-fly during training each time, a script has been added to build the HDF5 dataset once and save it to disk for faster loading during training. This can save a significant amount of time during training depending on your hardware.
- Additional flags to resume training from a checkpoint have been added (in this way you can continue training from a pre-trained model to check performance improvements over downstream tasks).
- The DeepSpeech library has been deprecated in favor of SpeechBrain for ASR evaluation using Wav2Vec2 models due to compatibility issues (mainly associated with CPU architecture incompatibilities).
- The CTC beam search decoder has been changed to use torchaudio's built-in decoder instead of the original ctcdecode library due to compatibility issues.
  - Additional instructions have been added to download the KenLM language model and generate the lexicon needed for decoding.
- Tensorboard logging has been added to monitor training progress.

### What could be improved further

There are some additional improvements that could be made to further improve usability:

- Following [Stanford - MONA LISA](https://github.com/tbenst/silent_speech), `hydra` could be used for configuration management to simplify hyperparameter tuning and experiment tracking. Additionally, `pytorch lightning` could be used to simplify the training loop and improve reproducibility.
- The dataset building process could be further optimized by using a more efficient data storage format or by implementing a more efficient data loading pipeline. Currently, the HDF5 dataset is built to be used with a single GPU setup.  Further work could be done to optimize for multi-GPU setups (e.g., using a different data sampler suitable for distributed training).
- Training progress could be monitored more easily using tools like `Weights & Biases` to store experiment results and visualize training metrics over time with less setup required.

## Data

The EMG and audio data can be downloaded from <https://doi.org/10.5281/zenodo.4064408>.  The scripts expect the data to be located in a `emg_data` subdirectory by default, but the location can be overridden with flags (see the data download script section below).

Force-aligned phonemes from the Montreal Forced Aligner have been included as a git submodule, which must be updated using the process described in "Environment Setup" below.
Note that there will not be an exception if the directory is not found, but logged phoneme prediction accuracies reporting 100% is a sign that the directory has not been loaded correctly.
>**Note**: A script to download the data directly into the expected directory structure is provided below.

## Environment Setup

The code has been tested with Python 3.10 and requires a number of Python packages, including PyTorch with CUDA support.
Set up a virtual environment and install the required packages. We suggest using [`uv`](https://docs.astral.sh/uv/) to install all the dependencies, including PyTorch with the appropriate CUDA version.

```bash
uv sync
```

alternatively, you can manually install the dependencies listed in `pyproject.toml`, making sure to install the correct version of PyTorch for your CUDA setup from <https://pytorch.org/get-started/locally/>.

You will also need to pull git submodules for Hifi-GAN and the phoneme alignment data, using the following commands:

```bash
git submodule init
git submodule update
tar -xvzf text_alignments/text_alignments.tar.gz
```

>**Note**: Due to compatibility issues, `DeepSpeech` library has been deprecated in favor of SpeechBrain for ASR evaluation using Wav2Vec2 models. Such models will be downloaded automatically when running the evaluation script and cached in your Hugging Face cache directory.

### Data Download Script

For convenience, a script is provided to download the data directly into the expected directory structure.
Before running the script, check the configuration files in the `config` directory to ensure the data paths are set correctly. You will need to change the `$DATA_PATH` variable to your desired data directory path or bind it to a valid path in your environment.
Then run:

```bash
python download_data.py
```

### Audio Cleaning

This is an optional step. Training will be faster if you re-run the audio cleaning, which will save re-sampled audio so it doesn't have to be re-sampled every training run.
In order to run the cleaning script, use the following command:

```bash
python data_collection/clean_audio.py
```

this script will run in parallel and may take a few minutes to complete, depending on your hardware. It will save cleaned audio files with the correct sample rate in the same directories as the original audio files, with filenames prefixed by `cleaned_`.
>**Note**: If you do not run this step, the code will re-sample the original audio files on-the-fly during training, which will require more time as the audio files will be resampled on CPU creating many CPU-GPU transfers during training.

### Building HDF5 Dataset

In the original code, the dataset was built on-the-fly during training each time, which can be time-consuming.
To build the HDF5 dataset from the raw EMG and audio files, run the following command, replacing the output file path as needed:

```bash
python build_hdf5.py
```

in this way, the dataset only needs to be built once and can be loaded quickly during training.

## Pre-trained Models

Pre-trained models for the vocoder and transduction model are available at
<https://doi.org/10.5281/zenodo.6747411>.

## Running

To train an EMG to speech feature transduction model, use the following command:

```bash
python transduction_model.py
```

all the training parameters can be adjusted in the `config/transduction_model.json` file. You can specify an output directory for saving models and logs in the configuration file. You can also start the training from a pre-trained model.

At the end of training, an ASR evaluation will be run on the validation set if a HiFi-GAN model checkpoint is provided.

To evaluate a model on the test set, use

```bash
python evaluate.py --model ./models/transduction_model/model.pt
```

test set file can be changed in the configuration file.

# Silent Speech Recognition

This section is about converting silent speech directly to text rather than synthesizing speech audio.
The speech-to-text model uses the same neural architecture but with a CTC decoder, and achieves a WER of approximately 28% (as described in the dissertation [Voicing Silent Speech](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2022/EECS-2022-68.pdf)).

Due to compatibility issues, the original ctcdecode library has been replaced with torchaudio's built-in CTC beam search decoder.

In order to use the CTC beam search decoder with a KenLM language model, you will need to download the KenLM language model as well as generating the lexicon inferred from the data.

## Setup

Downloading the KenLM language model can be done using the provided script in the KenLM subdirectory.

```bash
cd KenLM
python download_LM.py --output_directory .
```

in this way, the language model will be downloaded to the current directory under the `lm.bin` file.

After downloading the language model, you will need to generate the lexicon file used for decoding. The lexicon can be generated using the provided script `get_lexicon.py`:

```bash
python get_lexicon.py
```

## Running

Pre-trained model weights can be downloaded from <https://doi.org/10.5281/zenodo.7183877>.

To train a model, run

```bash
python recognition_model.py
```

To run a test set evaluation on a saved model, use

```bash
python recognition_model.py --evaluate_saved "./models/recognition_model/model.pt"
```
