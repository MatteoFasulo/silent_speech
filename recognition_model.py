import os
import sys
from datetime import datetime
import numpy as np
import logging
from torchaudio.models.decoder import ctc_decoder
import jiwer
import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

from hdf5_dataset import H5EmgDataset, SizeAwareSampler
from adapted_emg_transformer import EMGTransformer
from data_utils import combine_fixed_length, decollate_tensor

from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_boolean("debug", False, "debug")
flags.DEFINE_string("output_directory", "output", "where to save models and outputs")
flags.DEFINE_integer("batch_size", 32, "training batch size")
flags.DEFINE_integer("num_epochs", 200, "number of epochs")
flags.DEFINE_float("learning_rate", 3e-4, "learning rate")
flags.DEFINE_integer("learning_rate_warmup", 1000, "steps of linear warmup")
flags.DEFINE_float("l2", 0.0, "weight decay")
flags.DEFINE_string("start_training_from", None, "start training from this model")
flags.DEFINE_string("evaluate_saved", None, "run evaluation on given model file")
flags.DEFINE_integer("num_workers", 64, "number of workers for dataloaders")
flags.DEFINE_string("lm_directory", "/users/mfasulo/silent_speech/KenLM/", "directory with language model files")
flags.DEFINE_boolean("verbose", False, "print verbose output")

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
seq_len = 200


def test(model, dset, device):
    model.eval()

    tkns = [c for c in dset.text_transform.chars] + ["_"]
    decoder = ctc_decoder(
        lexicon=os.path.join(FLAGS.lm_directory, "gaddy_lexicon.txt"),
        tokens=tkns,
        lm=os.path.join(FLAGS.lm_directory, "lm.bin"),
        blank_token="_",
        sil_token="|",
        nbest=1,
        lm_weight=2,  # default is 2; Gaddy sets to 1.85
        # word_score  = -3,
        # sil_score   = -2,
        beam_size=150,  # SET TO 150 during inference
    )

    dataloader = torch.utils.data.DataLoader(
        dset,
        batch_size=1,
        pin_memory=(device == "cuda"),
        collate_fn=dset.collate_raw,
        num_workers=FLAGS.num_workers,
        persistent_workers=True,
    )

    references = []
    predictions = []

    with torch.no_grad():
        for example in tqdm.tqdm(dataloader, "Evaluate", disable=None):
            X = example["emg"][0].unsqueeze(0).to(device)
            X_raw = example["raw_emg"][0].unsqueeze(0).to(device)
            sess = example["session_ids"][0].to(device)

            pred = F.log_softmax(model(X, X_raw, sess), dim=-1)

            beam_results = decoder(pred.detach().cpu())
            pred_text = " ".join(beam_results[0][0].words).strip().lower()
            b0 = example["text"][0]
            target_text = dset.text_transform.clean_text(b0[0])

            if len(target_text) > 0:
                references.append(target_text)
                predictions.append(pred_text)

    if FLAGS.verbose:
        for ref, pred in zip(references, predictions):
            print(f"Ref: {ref}\nPred: {pred}\n")
            print("---" * 20)

    model.train()
    return jiwer.wer(references, predictions)


def train_model(model, trainset, devset, device):

    dataloader = torch.utils.data.DataLoader(
        trainset,
        pin_memory=(device == "cuda"),
        num_workers=FLAGS.num_workers,
        collate_fn=devset.collate_raw,
        batch_sampler=SizeAwareSampler(trainset, 128_000),
        persistent_workers=True,
    )

    n_chars = len(devset.text_transform.chars)
    if FLAGS.start_training_from is not None:
        state_dict = torch.load(FLAGS.start_training_from, map_location=device)
        # remove the w_out layer if it exists, as the shape would not match
        if "w_out.weight" in state_dict:
            del state_dict["w_out.weight"]
            del state_dict["w_out.bias"]
        model.load_state_dict(state_dict, strict=False)
        logging.info(f"Loaded model from {FLAGS.start_training_from}")

    optim = torch.optim.AdamW(model.parameters(), weight_decay=FLAGS.l2, lr=FLAGS.learning_rate)
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[125, 150, 175], gamma=0.5)

    def set_lr(new_lr):
        for param_group in optim.param_groups:
            param_group["lr"] = new_lr

    target_lr = FLAGS.learning_rate

    def schedule_lr(iteration):
        iteration = iteration + 1
        if iteration <= FLAGS.learning_rate_warmup:
            set_lr(iteration * target_lr / FLAGS.learning_rate_warmup)

    batch_idx = 0

    for epoch_idx in range(FLAGS.num_epochs):
        losses = []
        for example in tqdm.tqdm(dataloader, "Train step", disable=None):
            schedule_lr(batch_idx)

            X = combine_fixed_length(example["emg"], seq_len).to(device)
            X_raw = combine_fixed_length(example["raw_emg"], seq_len * 8).to(device)
            sess = combine_fixed_length(example["session_ids"], seq_len).to(device)

            pred = model(X, X_raw, sess)
            pred = F.log_softmax(pred, dim=-1)

            pred = nn.utils.rnn.pad_sequence(
                decollate_tensor(pred, example["lengths"]), batch_first=False
            )  # seq first, as required by ctc
            y = nn.utils.rnn.pad_sequence(example["text_int"], batch_first=True).to(device)
            loss = F.ctc_loss(pred, y, example["lengths"], example["text_int_lengths"], blank=n_chars)
            losses.append(loss.item())
            writer.add_scalar("train/loss_step", loss.item(), batch_idx)

            loss.backward()
            if (batch_idx + 1) % 2 == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 10)
                optim.step()
                optim.zero_grad(set_to_none=True)

            # del example, pred, loss, y, sess, X, X_raw
            # torch.cuda.empty_cache()

            batch_idx += 1

        train_loss = np.mean(losses)
        current_lr = optim.param_groups[0]["lr"]
        writer.add_scalar("train/loss_epoch", train_loss, epoch_idx)
        writer.add_scalar("train/lr", current_lr, epoch_idx)

        if (epoch_idx + 1) % 5 == 0:
            val = test(model, devset, device)
            writer.add_scalar("val/wer", val, epoch_idx)
            logging.info(
                f"finished epoch {epoch_idx+1} - training loss: {train_loss:.4f} validation WER: {val*100:.2f}"
            )
        else:
            logging.info(f"finished epoch {epoch_idx+1} - training loss: {train_loss:.4f} - no validation WER computed")

        lr_sched.step()

        torch.save(model.state_dict(), os.path.join(FLAGS.output_directory, "model.pt"))

    model.load_state_dict(torch.load(os.path.join(FLAGS.output_directory, "model.pt")))  # re-load best parameters

    return model


def evaluate_saved():
    device = "cuda" if torch.cuda.is_available() and not FLAGS.debug else "cpu"
    testset = H5EmgDataset(test=True)
    silent_flags = [d.silent for (d, _) in testset.example_indices]
    print(f"Unique silent flags in test set: {set(silent_flags)}")
    n_chars = len(testset.text_transform.chars)
    model = EMGTransformer(testset.num_features, n_chars + 1).to(device)
    model.load_state_dict(torch.load(FLAGS.evaluate_saved, map_location=torch.device(device)))
    summary(
        model,
        input_data=[
            torch.randn(1, FLAGS.img_size, FLAGS.in_chans).to(device),
            torch.randn(1, FLAGS.img_size, FLAGS.in_chans).to(device),
            torch.randn(1, FLAGS.img_size, FLAGS.in_chans).to(device),
        ],
    )
    print(f"Loaded model from {FLAGS.evaluate_saved}")
    test_wer = test(model, testset, device)
    print("WER:", test_wer)


def main():
    log_filename = os.path.join(FLAGS.output_directory, f"log.txt")
    os.makedirs(FLAGS.output_directory, exist_ok=True)
    logging.basicConfig(
        handlers=[logging.FileHandler(log_filename, "w"), logging.StreamHandler()],
        level=logging.INFO,
        format="%(message)s",
    )

    logging.info(sys.argv)

    trainset = H5EmgDataset(dev=False, test=False)
    devset = H5EmgDataset(dev=True)
    testset = H5EmgDataset(test=True)
    logging.info("output example: %s", devset.example_indices[0])
    logging.info("train / dev split: %d %d", len(trainset), len(devset))

    device = "cuda" if torch.cuda.is_available() and not FLAGS.debug else "cpu"

    n_chars = len(devset.text_transform.chars)
    model = EMGTransformer(devset.num_features, n_chars + 1).to(device)
    summary(
        model,
        input_data=[
            torch.randn(1, FLAGS.img_size, FLAGS.in_chans).to(device),
            torch.randn(1, FLAGS.img_size, FLAGS.in_chans).to(device),
            torch.randn(1, FLAGS.img_size, FLAGS.in_chans).to(device),
        ],
    )

    best_ckpt_model = train_model(model, trainset, devset, device)

    # Run test
    test_wer = test(best_ckpt_model, testset, device)
    logging.info("Test WER: %.2f%%", test_wer * 100)
    writer.add_scalar("test/wer", test_wer, 0)
    writer.add_hparams(
        {
            "window_size": FLAGS.img_size,
            "learning_rate": FLAGS.learning_rate,
            "l2": FLAGS.l2,
            "num_epochs": FLAGS.num_epochs,
            "embed_dim": FLAGS.embed_dim,
            "num_heads": FLAGS.num_heads,
            "num_layers": FLAGS.num_layers,
            "downsample_factor": FLAGS.downsample_factor,
            "mlp_ratio": FLAGS.mlp_ratio,
        },
        {
            "test_wer": test_wer,
        },
    )
    return


if __name__ == "__main__":
    FLAGS(sys.argv)
    if FLAGS.evaluate_saved is not None:
        evaluate_saved()
    else:
        global writer
        writer = SummaryWriter()
        main()
