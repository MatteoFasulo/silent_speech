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

```
python download_data.py --output_dir $SCRATCH/datasets/Gaddy/
```

## Environment Setup

```
uv venv -p 3.10 --relocatable --link-mode=copy .silentvenv
source .silentvenv/bin/activate
uv pip install --link-mode=copy torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
uv pip install --link-mode=copy absl-py numpy librosa pysoundfile matplotlib scipy numba unidecode tqdm jiwer==2.2.1 praat-textgrids noisereduce torchinfo tensorboard einops timm speechbrain h5py transformers 'huggingface_hub<1.0' omegaconf torch-tb-profiler pandas
python -m compileall -j 64 -o 0 -o 1 -o 2 $SCRATCH/.silentvenv/lib/python3.10/site-packages/
```

You will also need to pull git submodules for Hifi-GAN and the phoneme alignment data, using the following commands:
```
git submodule init
git submodule update
tar -xvzf text_alignments/text_alignments.tar.gz
```

Use the following commands to download pre-trained DeepSpeech model files for evaluation.  It is important that you use DeepSpeech version 0.7.0 model files for evaluation numbers to be consistent with the original papers.  Note that more recent DeepSpeech packages such as version 0.9.3 can be used as long as they are compatible with version 0.7.x model files.
```
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.7.0/deepspeech-0.7.0-models.pbmm
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.7.0/deepspeech-0.7.0-models.scorer
```

(Optional) Training will be faster if you re-run the audio cleaning, which will save re-sampled audio so it doesn't have to be re-sampled every training run.
```
python data_collection/clean_audio.py $SCRATCH/datasets/Gaddy/emg_data/nonparallel_data $SCRATCH/datasets/Gaddy/emg_data/silent_parallel_data $SCRATCH/datasets/Gaddy/emg_data/voiced_parallel_data
```

```
python build_hdf5.py --output_file /capstor/scratch/cscs/mfasulo/datasets/Gaddy/h5/emg_dataset.h5
```

## Pre-trained Models

Pre-trained models for the vocoder and transduction model are available at
<https://doi.org/10.5281/zenodo.6747411>.

## Running

To train an EMG to speech feature transduction model, use
```
python transduction_model.py --hifigan_checkpoint hifigan_finetuned/checkpoint --output_directory "./models/transduction_model/" --start_training_from /capstor/scratch/cscs/mfasulo/checkpoints/pretraining/20@rope@gelu/20@rope@gelu-epoch=49-val_loss=0.0091.ckpt --seed 42
```
where `hifigan_finetuned/checkpoint` is a trained HiFi-GAN generator model (optional).
At the end of training, an ASR evaluation will be run on the validation set if a HiFi-GAN model is provided.

To evaluate a model on the test set, use
```
python evaluate.py --models ./models/transduction_model/model.pt --hifigan_checkpoint hifigan_finetuned/checkpoint --output_directory evaluation_output --testset_file testset_origdev.json
```

Or to evaluate all the models in a directory, use
```bash
sh eval.sh ./models/transduction_model wer_results.csv
```

And to report mean and std dev of WER results from the resulting CSV file, use the following one-liner:
```python
python -c "import pandas as pd; df=pd.read_csv('wer_results.csv'); df['wer'] = pd.to_numeric(df['wer'], errors='coerce'); sliced_df=df[df['model'].str.contains('last')].dropna(); print(sliced_df); print(f'\nMean WER: {sliced_df[\"wer\"].mean():.4f}\nStd WER: {sliced_df[\"wer\"].std():.4f}')"
```

## HiFi-GAN Training

The HiFi-GAN model is fine-tuned from a multi-speaker model to the voice of this dataset.  Spectrograms predicted from the transduction model are used as input for fine-tuning instead of gold spectrograms.  To generate the files needed for HiFi-GAN fine-tuning, run the following with a trained model checkpoint:
```
python make_vocoder_trainset.py --model ./models/transduction_model/model.pt --output_directory hifigan_training_files
```
The resulting files can be used for fine-tuning using the instructions in the hifi-gan repository.
The pre-trained model was fine-tuned for 75,000 steps, starting from the `UNIVERSAL_V1` model provided by the HiFi-GAN repository.
Although the HiFi-GAN is technically fine-tuned for the output of a specific transduction model, we found it to transfer quite well and shared a single HiFi-GAN for most experiments.

```
cd hifi_gan
python train.py --fine_tuning True --config config_v1.json --input_wavs_dir ../hifigan_training_files/wavs --input_mels_dir ../hifigan_training_files/mels --input_training_file ../hifigan_training_files/train_filelist.txt --input_validation_file ../hifigan_training_files/dev_filelist.txt
```

# Silent Speech Recognition

This section is about converting silent speech directly to text rather than synthesizing speech audio.
The speech-to-text model uses the same neural architecture but with a CTC decoder, and achieves a WER of approximately 28% (as described in the dissertation [Voicing Silent Speech](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2022/EECS-2022-68.pdf)).

You will need to install the ctcdecode library (1.0.3) in addition to the libraries listed above to use the recognition code.
(This package cannot be built successfully under Windows platform)
```
pip install git+https://github.com/parlance/ctcdecode.git
```

And you will need to download a KenLM language model, such as this one from DeepSpeech:
```
curl https://github.com/mozilla/DeepSpeech/releases/download/v0.6.1/lm.binary
```

Pre-trained model weights can be downloaded from <https://doi.org/10.5281/zenodo.7183877>.

To train a model, run
```
python recognition_model.py --output_directory "./models/recognition_model/"
```

To run a test set evaluation on a saved model, use
```
python recognition_model.py --evaluate_saved "./models/recognition_model/model.pt"
```
