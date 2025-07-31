from copy import copy
import sys
import h5py
import json
import numpy as np
import pickle
import random
import os
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from data_utils import TextTransform, FeatureNormalizer
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_list(
    "silent_data_directories",
    ["/capstor/scratch/cscs/mfasulo/datasets/Gaddy/emg_data/silent_parallel_data/"],
    "silent data locations",
)
flags.DEFINE_list(
    "voiced_data_directories",
    [
        "/capstor/scratch/cscs/mfasulo/datasets/Gaddy/emg_data/voiced_parallel_data/",
        "/capstor/scratch/cscs/mfasulo/datasets/Gaddy/emg_data/nonparallel_data/",
    ],
    "voiced data locations",
)
flags.DEFINE_string("testset_file", "testset_largedev.json", "file with testset indices")
flags.DEFINE_string("h5_path", "/capstor/scratch/cscs/mfasulo/datasets/Gaddy/h5/emg_dataset.h5", "HDF5 file")


class EMGDirectory(object):
    def __init__(self, session_index, directory, silent, exclude_from_testset=False):
        self.session_index = session_index
        self.directory = directory
        self.name = os.path.basename(directory)
        self.silent = silent
        self.exclude_from_testset = exclude_from_testset

    def __lt__(self, other):
        return self.session_index < other.session_index

    def __repr__(self):
        return self.directory


class H5EmgDataset(Dataset):
    def __init__(self, dev=False, test=False, no_normalizers=False):
        super().__init__()

        # Set random seed for reproducible shuffling across processes
        random.seed(0)
        self.no_normalizers = no_normalizers

        # 1) load test/dev lists
        with open(FLAGS.testset_file) as f:
            split = json.load(f)
        dev_set = {tuple(x) for x in split["dev"]}
        test_set = {tuple(x) for x in split["test"]}

        # 2) build directories list
        dirs = []
        with h5py.File(FLAGS.h5_path, "r") as h5:
            # silent sessions
            if "silent" in h5:
                for sd in FLAGS.silent_data_directories:
                    for sess in h5["silent"]:
                        dirs.append(EMGDirectory(len(dirs), os.path.join(sd, sess), True))

            has_silent = len(FLAGS.silent_data_directories) > 0 and "silent" in h5 and len(h5["silent"]) > 0

            # voiced sessions
            if "voiced" in h5:
                # Create a map from session name to full path
                voiced_session_paths = {}
                for vd in FLAGS.voiced_data_directories:
                    if os.path.exists(vd):
                        for sess in os.listdir(vd):
                            if os.path.isdir(os.path.join(vd, sess)):
                                voiced_session_paths[sess] = os.path.join(vd, sess)

                for sess in h5["voiced"]:
                    if sess in voiced_session_paths:
                        dirs.append(
                            EMGDirectory(len(dirs), voiced_session_paths[sess], False, exclude_from_testset=has_silent)
                        )

        # 3) replicate example_indices logic
        example_indices = []
        with h5py.File(FLAGS.h5_path, "r") as h5:
            for d in dirs:
                mode = "silent" if d.silent else "voiced"
                if mode not in h5:
                    continue

                grp = h5[mode][d.name]
                for utt in grp:
                    g = grp[utt]
                    book = g.attrs["book"]
                    sent = int(g.attrs["sentence_index"])
                    if sent < 0:
                        continue
                    loc = (book, sent)
                    in_dev = loc in dev_set
                    in_test = loc in test_set

                    include = False
                    if test:
                        if in_test and not d.exclude_from_testset:
                            include = True
                    elif dev:
                        if in_dev and not d.exclude_from_testset:
                            include = True
                    else:  # train
                        if not in_test and not in_dev:
                            include = True

                    if include:
                        example_indices.append((d, utt))

        # preserve ordering and shuffle
        example_indices.sort()
        random.shuffle(example_indices)

        self.example_indices = example_indices
        self.text_transform = TextTransform()

        # Load normalizers only if not disabled
        if not self.no_normalizers:
            self.mfcc_norm, self.emg_norm = pickle.load(open(FLAGS.normalizers_file, "rb"))

        with h5py.File(FLAGS.h5_path, "r") as h5:
            d, utt = self.example_indices[0]
            mode = "silent" if d.silent else "voiced"
            grp = h5[mode][d.name][utt]
            sample_mfccs = grp["mfccs"][()]
            sample_emg = grp["emg_feats"][()]
            self.num_speech_features = sample_mfccs.shape[1]
            self.num_features = sample_emg.shape[1]

        self._h5 = None

    def subset(self, fraction):
        result = copy(self)
        result.example_indices = self.example_indices[: int(fraction * len(self.example_indices))]
        return result

    def __len__(self):
        return len(self.example_indices)

    def __getitem__(self, i):
        if self._h5 is None:
            self._h5 = h5py.File(FLAGS.h5_path, "r", swmr=True)

        d, utt = self.example_indices[i]
        grp = self._h5["silent" if d.silent else "voiced"][d.name][utt]

        # load arrays + metadata
        mfccs = grp["mfccs"][()]
        emg_feats = grp["emg_feats"][()]
        raw_emg = grp["raw_emg"][()]
        phonemes = grp["phonemes"][()]
        text = grp.attrs["text"]
        silent = d.silent

        # Apply normalization only if not disabled
        if not self.no_normalizers:
            mfccs = self.mfcc_norm.normalize(mfccs)
            emg = self.emg_norm.normalize(emg_feats)
            emg = 8 * np.tanh(emg / 8.0)
        else:
            emg = emg_feats

        # Always apply raw_emg transformations (not part of normalizers)
        raw = raw_emg / 20.0
        raw = 50 * np.tanh(raw / 50.0)

        sample = {
            "audio_features": torch.from_numpy(mfccs),
            "emg": torch.from_numpy(emg),
            "raw_emg": torch.from_numpy(raw),
            "phonemes": torch.from_numpy(phonemes),
            "text": text,
            "text_int": torch.from_numpy(np.array(self.text_transform.text_to_int(text), dtype=np.int64)),
            "text_int_lengths": len(text),
            "session_ids": torch.full((emg.shape[0],), fill_value=d.session_index, dtype=torch.int64),
            "book_location": (grp.attrs["book"], int(grp.attrs["sentence_index"])),
            "silent": silent,
        }

        if d.silent:
            # parallel voiced data for silent sessions
            voiced_emg = grp["parallel_voiced_emg"][()]
            voiced_mfccs = grp["parallel_voiced_audio_features"][()]

            # Apply normalization to parallel data if not disabled
            if not self.no_normalizers:
                voiced_mfccs = self.mfcc_norm.normalize(voiced_mfccs)
                voiced_emg = self.emg_norm.normalize(voiced_emg)
                voiced_emg = 8 * np.tanh(voiced_emg / 8.0)

            sample["parallel_voiced_audio_features"] = torch.from_numpy(voiced_mfccs)
            sample["parallel_voiced_emg"] = torch.from_numpy(voiced_emg)

        return sample

    @staticmethod
    def collate_raw(batch):
        batch_size = len(batch)
        audio_features = []
        audio_feature_lengths = []
        parallel_emg = []
        for ex in batch:
            if ex["silent"]:
                audio_features.append(ex["parallel_voiced_audio_features"])
                audio_feature_lengths.append(ex["parallel_voiced_audio_features"].shape[0])
                parallel_emg.append(ex["parallel_voiced_emg"])
            else:
                audio_features.append(ex["audio_features"])
                audio_feature_lengths.append(ex["audio_features"].shape[0])
                parallel_emg.append(np.zeros(1))
        phonemes = [ex["phonemes"] for ex in batch]
        emg = [ex["emg"] for ex in batch]
        raw_emg = [ex["raw_emg"] for ex in batch]
        session_ids = [ex["session_ids"] for ex in batch]
        lengths = [ex["emg"].shape[0] for ex in batch]
        silent = [ex["silent"] for ex in batch]
        text_ints = [ex["text_int"] for ex in batch]
        text_lengths = [ex["text_int"].shape[0] for ex in batch]
        text = [ex["text"] for ex in batch]

        result = {
            "audio_features": audio_features,
            "audio_feature_lengths": audio_feature_lengths,
            "emg": emg,
            "raw_emg": raw_emg,
            "parallel_voiced_emg": parallel_emg,
            "phonemes": phonemes,
            "session_ids": session_ids,
            "lengths": lengths,
            "silent": silent,
            "text_int": text_ints,
            "text_int_lengths": text_lengths,
            "text": text,
        }
        return result


class SizeAwareSampler(torch.utils.data.Sampler):
    def __init__(self, emg_dataset, max_len):
        self.dataset = emg_dataset
        self.max_len = max_len
        # ensure HDF5 is open
        if self.dataset._h5 is None:
            # open in read‑only SWMR mode
            self.dataset._h5 = h5py.File(FLAGS.h5_path, "r", swmr=True)
        self.batches = self._create_batches()

    def _create_batches(self):
        batches, batch, batch_len = [], [], 0
        for idx in random.sample(range(len(self.dataset)), len(self.dataset)):
            # grab the length from HDF5
            d, utt = self.dataset.example_indices[idx]
            mode = "silent" if d.silent else "voiced"
            sess = os.path.basename(d.directory)
            grp = self.dataset._h5[mode][sess][utt]
            length = grp.attrs["raw_emg_len"]

            if batch and batch_len + length > self.max_len:
                batches.append(batch)
                batch, batch_len = [], 0

            batch.append(idx)
            batch_len += length

        if batch:
            batches.append(batch)
        return batches

    def __iter__(self):
        random.shuffle(self.batches)
        yield from self.batches

    def __len__(self):
        return len(self.batches)
