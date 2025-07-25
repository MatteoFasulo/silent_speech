import sys
import os
import logging

import tqdm

import torch
from torch import nn

#from architecture import Model
from models import NewModel
from transduction_model import test, save_output
#from read_emg import EMGDataset
from hdf5_dataset import H5EmgDataset
from asr_evaluation import evaluate
from data_utils import phoneme_inventory, print_confusion
from vocoder import Vocoder

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_list('models', [], 'identifiers of models to evaluate')

class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x_raw):
        ys = []
        ps = []
        for model in self.models:
            y, p = model(x_raw)
            ys.append(y)
            ps.append(p)
        return torch.stack(ys,0).mean(0), torch.stack(ps,0).mean(0)

def main():
    os.makedirs(FLAGS.output_directory, exist_ok=True)
    logging.basicConfig(handlers=[
            logging.FileHandler(os.path.join(FLAGS.output_directory, 'eval_log.txt'), 'w'),
            logging.StreamHandler()
            ], level=logging.INFO, format="%(message)s")

    testset = H5EmgDataset(dev=False, test=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    models = []
    for fname in FLAGS.models:
        state_dict = torch.load(fname, map_location=device, weights_only=True)
        model = NewModel(testset.num_speech_features, len(phoneme_inventory)).to(device)
        model.load_state_dict(state_dict, strict=False)
        models.append(model)
    ensemble = EnsembleModel(models)

    _, _, confusion = test(ensemble, testset, device)
    print_confusion(confusion)

    vocoder = Vocoder()

    for i, datapoint in enumerate(tqdm.tqdm(testset, 'Generate outputs', disable=None)):
        save_output(ensemble, datapoint, os.path.join(FLAGS.output_directory, f'example_output_{i}.wav'), device, testset.mfcc_norm, vocoder)

    evaluate(testset, FLAGS.output_directory)

if __name__ == "__main__":
    FLAGS(sys.argv)
    main()
