# build the lexicon from the dataset
import sys
import tqdm
from hdf5_dataset import H5EmgDataset
from data_utils import TextTransform

from absl import flags

FLAGS = flags.FLAGS

transform = TextTransform()


def get_unigram(dataset):
    unigram = set()
    for example in tqdm.tqdm(dataset, "Building unigram"):
        clean = transform.clean_text(example["text"])
        for w in clean.split():
            unigram.add(w)
    return unigram


def get_lexicon(vocab):
    with open("KenLM/gaddy_lexicon.txt", "w") as fout:
        for word in vocab:
            # split word into char tokens:
            chars = list(word)
            fout.write(f"{word} " + " ".join(chars) + " |\n")


if __name__ == "__main__":
    FLAGS(sys.argv)
    trainset = H5EmgDataset(dev=False, test=False)
    devset = H5EmgDataset(dev=True, test=False)
    testset = H5EmgDataset(dev=False, test=True)

    train_unigram = get_unigram(trainset)
    dev_unigram = get_unigram(devset)
    test_unigram = get_unigram(testset)

    merged_unigram = train_unigram | dev_unigram | test_unigram

    get_lexicon(merged_unigram)
    print(f"Lexicon saved to gaddy_lexicon.txt with {len(merged_unigram)} unique words.")
