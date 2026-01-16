# build the lexicon from the dataset
import argparse

import tqdm

from data_utils import TextTransform
from hdf5_dataset import H5EmgDataset

transform = TextTransform()


def get_unigram(dataset):
    unigram = set()
    for example in tqdm.tqdm(dataset, "Building unigram"):
        clean = transform.clean_text(example["text"])
        for w in clean.split():
            unigram.add(w)
    return unigram


def get_lexicon(vocab, output_file="gaddy_lexicon.txt"):
    with open(output_file, "w", encoding="utf-8") as fout:
        for word in vocab:
            # split word into char tokens:
            chars = list(word)
            fout.write(f"{word} " + " ".join(chars) + " |\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Build lexicon from dataset")
    parser.add_argument(
        "--output_file",
        type=str,
        default="gaddy_lexicon.txt",
        help="Output lexicon file",
    )
    args = parser.parse_args()
    trainset = H5EmgDataset(dev=False, test=False)
    devset = H5EmgDataset(dev=True, test=False)
    testset = H5EmgDataset(dev=False, test=True)

    train_unigram = get_unigram(trainset)
    dev_unigram = get_unigram(devset)
    test_unigram = get_unigram(testset)

    merged_unigram = train_unigram | dev_unigram | test_unigram

    get_lexicon(merged_unigram, output_file=args.output_file)
    print(
        f"Lexicon saved to {args.output_file} with {len(merged_unigram)} unique words."
    )
