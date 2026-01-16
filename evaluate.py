import argparse
import logging
import os
import random

import numpy as np
import torch
import tqdm

from architecture import EMGTransformer
from asr_evaluation import evaluate
from data_utils import load_config, phoneme_inventory, print_confusion
from hdf5_dataset import H5EmgDataset
from transduction_model import save_output, test
from vocoder import Vocoder

FLAGS = load_config(os.path.join("config", "transduction_model.json"))


def main():
    os.makedirs(FLAGS.output_directory, exist_ok=True)
    logging.basicConfig(
        handlers=[
            logging.FileHandler(os.path.join(FLAGS.output_directory, "eval_log.txt"), "w"),
            logging.StreamHandler(),
        ],
        level=logging.INFO,
        format="%(message)s",
    )

    dev = FLAGS.dev
    testset = H5EmgDataset(dev=dev, test=not dev)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    state_dict = torch.load(FLAGS.model, map_location=device)
    model = EMGTransformer(testset.num_features, testset.num_speech_features, len(phoneme_inventory)).to(device)
    model.load_state_dict(state_dict, strict=True)  # Ensure all keys match

    _, _, confusion = test(model, testset, device)
    # Save the confusion matrix
    np.save(os.path.join(FLAGS.output_directory, "confusion_matrix.npy"), confusion)
    print_confusion(confusion)

    vocoder = Vocoder()

    for i, datapoint in enumerate(tqdm.tqdm(testset, "Generate outputs", disable=None)):
        save_output(
            model,
            datapoint,
            os.path.join(FLAGS.output_directory, f"example_output_{i}.wav"),
            device,
            testset.mfcc_norm,
            vocoder,
        )

    evaluate(testset, FLAGS.output_directory)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Evaluate saved model on test set")
    args.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the saved model checkpoint",
    )
    args.add_argument(
        "--output_directory",
        type=str,
        default="evaluation_output",
        help="Directory to save evaluation outputs",
    )
    arg = args.parse_args()
    FLAGS.model = arg.model
    FLAGS.output_directory = arg.output_directory
    torch.manual_seed(FLAGS.seed)
    torch.cuda.manual_seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    main()
