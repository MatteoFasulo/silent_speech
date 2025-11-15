import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy
from absl import flags
from joblib import Parallel, delayed
from tqdm import tqdm

from data_utils import (get_emg_features, load_audio, phoneme_inventory,
                        read_phonemes)

FLAGS = flags.FLAGS
# flags.DEFINE_list('remove_channels', [], 'channels to remove')
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
flags.DEFINE_string(
    "testset_file", "testset_largedev.json", "file with testset indices"
)
flags.DEFINE_string(
    "text_align_directory", "text_alignments", "directory with alignment files"
)
flags.DEFINE_string("output_file", "emg_dataset.h5", "output HDF5 file name")


def remove_drift(signal, fs):
    b, a = scipy.signal.butter(3, 2, "highpass", fs=fs)
    return scipy.signal.filtfilt(b, a, signal)


def notch(signal, freq, sample_frequency):
    b, a = scipy.signal.iirnotch(freq, 30, sample_frequency)
    return scipy.signal.filtfilt(b, a, signal)


def notch_harmonics(signal, freq, sample_frequency):
    for harmonic in range(1, 8):
        signal = notch(signal, freq * harmonic, sample_frequency)
    return signal


def subsample(signal, new_freq, old_freq):
    times = np.arange(len(signal)) / old_freq
    sample_times = np.arange(0, times[-1], 1 / new_freq)
    result = np.interp(sample_times, times, signal)
    return result


def apply_to_all(function, signal_array, *args, **kwargs):
    results = []
    for i in range(signal_array.shape[1]):
        results.append(function(signal_array[:, i], *args, **kwargs))
    return np.stack(results, 1)


def load_utterance(
    base_dir, index, limit_length=False, debug=False, text_align_directory=None
):
    index = int(index)
    raw_emg = np.load(os.path.join(base_dir, f"{index}_emg.npy"))
    before = os.path.join(base_dir, f"{index-1}_emg.npy")
    after = os.path.join(base_dir, f"{index+1}_emg.npy")
    if os.path.exists(before):
        raw_emg_before = np.load(before)
    else:
        raw_emg_before = np.zeros([0, raw_emg.shape[1]])
    if os.path.exists(after):
        raw_emg_after = np.load(after)
    else:
        raw_emg_after = np.zeros([0, raw_emg.shape[1]])

    x = np.concatenate([raw_emg_before, raw_emg, raw_emg_after], 0)
    x = apply_to_all(notch_harmonics, x, 60, 1000)
    x = apply_to_all(remove_drift, x, 1000)
    x = x[raw_emg_before.shape[0] : x.shape[0] - raw_emg_after.shape[0], :]
    emg_orig = apply_to_all(subsample, x, 689.06, 1000)
    x = apply_to_all(subsample, x, 516.79, 1000)
    emg = x

    # for c in FLAGS.remove_channels:
    #    emg[:,int(c)] = 0
    #    emg_orig[:,int(c)] = 0

    emg_features = get_emg_features(emg)

    mfccs = load_audio(
        os.path.join(base_dir, f"{index}_audio_resampled.flac"),
        max_frames=min(emg_features.shape[0], 800 if limit_length else float("inf")),
    )

    if emg_features.shape[0] > mfccs.shape[0]:
        emg_features = emg_features[: mfccs.shape[0], :]
    assert emg_features.shape[0] == mfccs.shape[0]
    emg = emg[6 : 6 + 6 * emg_features.shape[0], :]
    emg_orig = emg_orig[8 : 8 + 8 * emg_features.shape[0], :]
    assert emg.shape[0] == emg_features.shape[0] * 6

    with open(os.path.join(base_dir, f"{index}_info.json")) as f:
        info = json.load(f)

    sess = os.path.basename(base_dir)
    tg_fname = f"{text_align_directory}/{sess}/{sess}_{index}_audio.TextGrid"
    if os.path.exists(tg_fname):
        phonemes = read_phonemes(tg_fname, mfccs.shape[0])
    else:
        phonemes = np.zeros(mfccs.shape[0], dtype=np.int64) + phoneme_inventory.index(
            "sil"
        )

    return (
        mfccs,
        emg_features,
        info["text"],
        (info["book"], info["sentence_index"]),
        phonemes,
        emg_orig.astype(np.float32),
    )


def gather_utterance_records(mode, base_dir):
    """Return a list of dicts, one per utterance in this directory."""
    records = []
    for fname in os.listdir(base_dir):
        if not fname.endswith("_info.json"):
            continue
        idx = fname.split("_")[0]
        mfccs, emg_feats, text, loc, phonemes, raw_emg = load_utterance(
            base_dir, idx, limit_length=False, text_align_directory=TEXT_ALIGN_DIR
        )
        with open(os.path.join(base_dir, f"{idx}_info.json")) as f:
            info = json.load(f)
        raw_emg_len = sum(c[0] for c in info["chunks"])
        rec = {
            "mode": mode,
            "session": os.path.basename(base_dir),
            "utt": idx,
            "mfccs": mfccs,
            "emg_feats": emg_feats,
            "raw_emg": raw_emg,
            "raw_emg_len": raw_emg_len,
            "phonemes": phonemes,
            "text": text,
            "book": loc[0],
            "sentence_index": loc[1],
            "silent": mode == "silent",
        }
        records.append(rec)
    return records


def main():
    # 1) build list of (mode,dir) pairs
    tasks = []
    for sd in SILENT_DIRS:
        for sess in sorted(os.listdir(sd)):
            tasks.append(("silent", os.path.join(sd, sess)))
    for vd in VOICED_DIRS:
        for sess in sorted(os.listdir(vd)):
            tasks.append(("voiced", os.path.join(vd, sess)))

    # 2) parallel load into Python structures
    all_records = Parallel(n_jobs=24, verbose=1)(
        delayed(gather_utterance_records)(mode, d) for mode, d in tasks
    )
    # flatten list of lists
    all_records = [r for rec_list in all_records for r in rec_list]
    print(f"Loaded {len(all_records)} utterances into RAM.")

    # 3) Create a map of voiced data locations
    print("Building voiced data map...")
    voiced_data_map = {}
    for rec in all_records:
        if not rec["silent"]:
            location = (rec["book"], rec["sentence_index"])
            if location[1] >= 0:  # Ensure we only map valid sentences
                voiced_data_map[location] = rec

    # 4) Link silent records to their voiced counterparts
    print("Linking silent records to voiced counterparts...")
    for rec in tqdm(all_records, desc="Linking data"):
        if rec["silent"]:
            location = (rec["book"], rec["sentence_index"])
            if location in voiced_data_map:
                voiced_rec = voiced_data_map[location]
                rec["parallel_voiced_audio_features"] = voiced_rec["mfccs"]
                rec["parallel_voiced_emg"] = voiced_rec["emg_feats"]
                rec["phonemes"] = voiced_rec["phonemes"]
            # else:
            # Optionally, you could add a warning here if a silent utterance
            # has no voiced parallel, though the original code doesn't.

    # 5) single HDF5 write
    with h5py.File(OUT_FILE, "w") as h5:
        for rec in tqdm(all_records, desc="Writing to HDF5"):
            # Skip utterances that don't have a valid sentence index
            if rec["sentence_index"] < 0:
                continue

            grp = h5.require_group(f"{rec['mode']}/{rec['session']}")
            utt_grp = grp.create_group(rec["utt"])

            utt_grp.create_dataset("mfccs", data=rec["mfccs"], chunks=True)
            utt_grp.create_dataset("emg_feats", data=rec["emg_feats"], chunks=True)
            utt_grp.create_dataset("raw_emg", data=rec["raw_emg"], chunks=True)
            utt_grp.create_dataset("phonemes", data=rec["phonemes"], chunks=True)
            utt_grp.attrs["raw_emg_len"] = rec["raw_emg_len"]

            # If it's a silent utterance, also save the parallel data
            if rec["silent"] and "parallel_voiced_audio_features" in rec:
                utt_grp.create_dataset(
                    "parallel_voiced_audio_features",
                    data=rec["parallel_voiced_audio_features"],
                    chunks=True,
                )
                utt_grp.create_dataset(
                    "parallel_voiced_emg", data=rec["parallel_voiced_emg"], chunks=True
                )

            utt_grp.attrs["text"] = rec["text"]
            utt_grp.attrs["book"] = rec["book"]
            utt_grp.attrs["sentence_index"] = rec["sentence_index"]
            utt_grp.attrs["silent"] = rec["silent"]

    print("All done—HDF5 file written:", OUT_FILE)


if __name__ == "__main__":
    FLAGS(sys.argv)
    SILENT_DIRS = FLAGS.silent_data_directories
    VOICED_DIRS = FLAGS.voiced_data_directories
    TEXT_ALIGN_DIR = FLAGS.text_align_directory
    out_path = Path(FLAGS.output_file).absolute()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE = str(out_path)
    main()
