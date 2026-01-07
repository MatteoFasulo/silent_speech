# Voicing Silent Speech

This repository contains code for synthesizing speech audio from silently mouthed words captured with electromyography (EMG).
It is the official repository for the papers [Digital Voicing of Silent Speech](https://aclanthology.org/2020.emnlp-main.445.pdf) at EMNLP 2020, [An Improved Model for Voicing Silent Speech](https://aclanthology.org/2021.acl-short.23.pdf) at ACL 2021, and the dissertation [Voicing Silent Speech](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2022/EECS-2022-68.pdf).
The current commit contains only the most recent model, but the versions from prior papers can be found in the commit history.
On an ASR-based open vocabulary evaluation, the latest model achieves a WER of approximately 36%.
Audio samples can be found [here](https://dgaddy.github.io/silent_speech_samples/June2022/).

The repository also includes code for directly converting silent speech to text.  See the section labeled [Silent Speech Recognition](#silent-speech-recognition).

## Data

The EMG and audio data can be downloaded from <https://doi.org/10.5281/zenodo.4064408>.  The scripts expect the data to be located in a `emg_data` subdirectory by default, but the location can be overridden with flags (see the top of `read_emg.py`).

Force-aligned phonemes from the Montreal Forced Aligner have been included as a git submodule, which must be updated using the process described in "Environment Setup" below.
Note that there will not be an exception if the directory is not found, but logged phoneme prediction accuracies reporting 100% is a sign that the directory has not been loaded correctly.

### Data Download Script

For convenience, a script is provided to download the data directly into the expected directory structure.
Before running the script, set the `$DATA_PATH` environment variable to point to your data directory.
Then run:

```bash
python download_data.py --output_dir $DATA_PATH/datasets/Gaddy/
```

## Environment Setup

The code has been tested with Python 3.10.10 and requires a number of Python packages, including PyTorch with CUDA support.
Set up a virtual environment and install the required packages using the following commands:

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install absl-py numpy librosa pysoundfile matplotlib scipy numba unidecode tqdm jiwer==2.2.1 praat-textgrids noisereduce torchinfo tensorboard einops timm speechbrain h5py transformers 'huggingface_hub<1.0' omegaconf torch-tb-profiler pandas flashlight-text KenLM
```

You will also need to pull git submodules for Hifi-GAN and the phoneme alignment data, using the following commands:

```bash
git submodule init
git submodule update
tar -xvzf text_alignments/text_alignments.tar.gz
```

Due to compatibility issues, `DeepSpeech` library has been deprecated in favor of SpeechBrain for ASR evaluation using Wav2Vec2 models.

### Audio Cleaning

This is an optional step. Training will be faster if you re-run the audio cleaning, which will save re-sampled audio so it doesn't have to be re-sampled every training run.
In order to run the cleaning script, use the following command, replacing `$DATA_PATH` with your data directory path:

```bash
python data_collection/clean_audio.py $DATA_PATH/datasets/Gaddy/emg_data/nonparallel_data $DATA_PATH/datasets/Gaddy/emg_data/silent_parallel_data $DATA_PATH/datasets/Gaddy/emg_data/voiced_parallel_data
```

this script will run in parallel and may take a few minutes to complete, depending on your hardware. It will save cleaned audio files with the correct sample rate in the same directories as the original audio files, with filenames prefixed by `cleaned_`.

### Building HDF5 Dataset

To build the HDF5 dataset from the raw EMG and audio files, run the following command, replacing the output file path as needed:

```bash
python build_hdf5.py --output_file $DATA_PATH/datasets/Gaddy/h5/emg_dataset.h5
```

in this way, the dataset only needs to be built once avoiding the original code that built the dataset on-the-fly during training each time.

## Pre-trained Models

Pre-trained models for the vocoder and transduction model are available at
<https://doi.org/10.5281/zenodo.6747411>.

## Running

To train an EMG to speech feature transduction model, use the following command:

```bash
python transduction_model.py --hifigan_checkpoint hifigan_finetuned/checkpoint --output_directory "./models/transduction_model/" --start_training_from <your_checkpoint_path> --seed 42
```

where `hifigan_finetuned/checkpoint` is a trained HiFi-GAN generator model (optional).
>Note: The `--start_training_from` flag is optional and it was added to continue training from a pre-trained model. You can remove this flag to train from scratch.

At the end of training, an ASR evaluation will be run on the validation set if a HiFi-GAN model is provided.

To evaluate a model on the test set, use

```bash
python evaluate.py --models ./models/transduction_model/model.pt --hifigan_checkpoint hifigan_finetuned/checkpoint --output_directory evaluation_output --testset_file testset_origdev.json
```

>Note: The `--testset_file` flag is set to `testset_origdev.json` by default, which will run the evaluation on the original development set.

A shell script is also provided to simplify evaluation. To evaluate all the models in a directory, use

```bash
sh eval.sh ./models/transduction_model wer_results.csv
```

which will save WER results for each model in the specified directory to `wer_results.csv`.

Then, the following one-liner can be used to print the results from the CSV file using pandas to report mean and std dev of WER results:

```python
python -c "import pandas as pd; df=pd.read_csv('wer_results.csv'); df['wer'] = pd.to_numeric(df['wer'], errors='coerce'); sliced_df=df[df['model'].str.contains('last')].dropna(); print(sliced_df); print(f'\nMean WER: {sliced_df[\"wer\"].mean():.4f}\nStd WER: {sliced_df[\"wer\"].std():.4f}')"
```

## HiFi-GAN Training

The HiFi-GAN model is fine-tuned from a multi-speaker model to the voice of this dataset.  Spectrograms predicted from the transduction model are used as input for fine-tuning instead of gold spectrograms. To generate the files needed for HiFi-GAN fine-tuning, run the following with a trained model checkpoint:

```bash
python make_vocoder_trainset.py --model ./models/transduction_model/model.pt --output_directory hifigan_training_files
```

The resulting files can be used for fine-tuning using the instructions in the hifi-gan repository.
The pre-trained model was fine-tuned for 75,000 steps, starting from the `UNIVERSAL_V1` model provided by the HiFi-GAN repository.
Although the HiFi-GAN is technically fine-tuned for the output of a specific transduction model, we found it to transfer quite well and shared a single HiFi-GAN for most experiments.
>Note: In TinyMyo, the HiFi-GAN checkpoint used was the same provided in the repository without further fine-tuning.

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

The lexicon can be generated using the provided script `get_lexicon.py`:

```bash
python get_lexicon.py
```

## Running

Pre-trained model weights can be downloaded from <https://doi.org/10.5281/zenodo.7183877>.

To train a model, run

```
python recognition_model.py --output_directory "./models/recognition_model/"
```

To run a test set evaluation on a saved model, use

```
python recognition_model.py --evaluate_saved "./models/recognition_model/model.pt"
```
