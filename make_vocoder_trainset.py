import os
import sys

import librosa
import numpy as np
import soundfile as sf
import torch
from absl import flags
from tqdm import tqdm

from architecture import EMGTransformer
from data_utils import phoneme_inventory
from hdf5_dataset import H5EmgDataset as EMGDataset
from transduction_model import get_aligned_prediction

FLAGS = flags.FLAGS
flags.DEFINE_string("model", None, "checkpoint of model to run")


def main():
    trainset = EMGDataset(dev=False, test=False)
    devset = EMGDataset(dev=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_phones = len(phoneme_inventory)
    model = EMGTransformer(devset.num_features, devset.num_speech_features, n_phones).to(device)
    state_dict = torch.load(FLAGS.model)
    model.load_state_dict(state_dict)

    os.makedirs(os.path.join(FLAGS.output_directory, "mels"), exist_ok=True)
    os.makedirs(os.path.join(FLAGS.output_directory, "wavs"), exist_ok=True)

    for dataset, name_prefix in [(trainset, "train"), (devset, "dev")]:
        with open(os.path.join(FLAGS.output_directory, f"{name_prefix}_filelist.txt"), "w") as filelist:
            for i, datapoint in enumerate(tqdm(dataset, desc=f"Processing {name_prefix} set")):
                spec = get_aligned_prediction(model, datapoint, device, dataset.mfcc_norm)
                spec = spec.T[np.newaxis, :, :].detach().cpu().numpy()
                np.save(
                    os.path.join(FLAGS.output_directory, "mels", f"{name_prefix}_output_{i}.npy"),
                    spec,
                )
                audio, r = sf.read(datapoint["audio_file"])
                if r != 22050:
                    audio = librosa.resample(audio, orig_sr=r, target_sr=22050, res_type="kaiser_fast")
                audio = np.clip(audio, -1, 1)  # because resampling sometimes pushes things out of range
                sf.write(
                    os.path.join(FLAGS.output_directory, "wavs", f"{name_prefix}_output_{i}.wav"),
                    audio,
                    22050,
                )
                filelist.write(f"{name_prefix}_output_{i}\n")


if __name__ == "__main__":
    FLAGS(sys.argv)
    main()
