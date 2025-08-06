import sys

import matplotlib.pyplot as plt
import pandas as pd
import tqdm
import torch

from hdf5_dataset import H5EmgDataset, SizeAwareSampler

from absl import flags
FLAGS = flags.FLAGS

if __name__ == "__main__":
    FLAGS(sys.argv)

    trainset = H5EmgDataset(dev=False, test=False)
    train_dloader = torch.utils.data.DataLoader(
        trainset,
        collate_fn=trainset.collate_raw,
        batch_sampler=SizeAwareSampler(trainset, 128_000),
    )

    classes_cnt = {
        'unvoiced': 0,
        'voiced': 0,
    }
    props = []
    for batch in tqdm.tqdm(train_dloader):
        silent_dist = batch['silent']
        n_samples = len(silent_dist)
        how_many_silent = silent_dist.count(True)

        classes_cnt['unvoiced'] += how_many_silent
        classes_cnt['voiced'] += n_samples - how_many_silent

        silent_prop = how_many_silent / n_samples
        silent_prop = round(silent_prop, 2)
        props.append(silent_prop * 100)

    print(f"Classes count: {classes_cnt}")

    props = pd.Series(props)
    props.to_csv("silent_proportions_batch.csv", index=False)
