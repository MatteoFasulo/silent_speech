# Voicing Silent Speech

Code for silent-speech recognition and EMG-to-audio transduction, based on [Gaddy's Silent Speech repository](https://github.com/dgaddy/silent_speech) and adapted to TinyMyo EMG model.

## Setup

```bash
uv sync
git submodule update --init
```

Set `DATA_PATH` and `CKPT_DIR` in the environment. Dataset paths and the train/dev/test split are defined in `config/data.yaml`; both training scripts save checkpoints under `CKPT_DIR`.

```bash
export DATA_PATH=/path/to/data
export CKPT_DIR=/path/to/checkpoints
```

Extract the alignment archive when needed:

```bash
tar -xvzf text_alignments/text_alignments.tar.gz
```

Prepare the data:

```bash
python download_data.py # downloads the EMG dataset
python data_collection/clean_audio.py # resamples audio
python build_hdf5.py # builds HDF5 files for training
```

## Configuration

Configuration uses OmegaConf:

- `config/data.yaml`: shared dataset settings
- `config/recognition.yaml`: recognition settings
- `config/transduction.yaml`: transduction settings

Command-line values override the configuration, including boolean flags:

```bash
python recognition_model.py --model tinymyo --verbose
python recognition_model.py --model tinymyo --num_epochs 200
```

## Transduction (EMG-to-audio)

```bash
python transduction_model.py
```

The optional `hifigan_checkpoint` setting enables waveform generation during training and evaluation. A pretrained transduction model and HiFi-GAN checkpoint are available from the [transduction-model Zenodo record](https://doi.org/10.5281/zenodo.6747411).

Evaluate a saved model:

```bash
python transduction_model.py \
  --evaluate_saved path/to/checkpoint.pt \
  --output_dir path/to/evaluation
```

## Recognition (EMG-to-text)

Train TinyMyo architecture for silent-speech recognition:

```bash
python recognition_model.py --model tinymyo
```

Checkpoints are saved after every epoch in:

```text
<checkpoint_directory>/<model>/<run_id>/epoch=NNN.pt
```

Evaluate a checkpoint:

```bash
python recognition_model.py \
  --model tinymyo \
  --evaluate_saved path/to/checkpoint.pt
```

Pretrained recognition checkpoints are available from the [recognition-model Zenodo record](https://doi.org/10.5281/zenodo.7183877). Checkpoint architecture and configuration must match (`tinymyo` or `gaddy`).

## Decoder

The current decoder uses the torchaudio/Flashlight CTC decoder with KenLM.
Required files are:

```text
KenLM/lm.bin
KenLM/gaddy_derived_lexicon.txt
```

Download the KenLM language model with:

```bash
python download_kenlm.py --output_dir KenLM
```

`gaddy_derived_lexicon.txt` is tracked in the repository and is the default lexicon for the current decoder.

## References

- [EMG dataset](https://doi.org/10.5281/zenodo.4064408)
- [Original repository](https://github.com/dgaddy/silent_speech)
