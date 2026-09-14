import logging
import os
import sys
from datetime import datetime

import jiwer
import numpy as np
import torch
import torch.nn.functional as F
import torchprofile
import tqdm
from safetensors.torch import load_file as load_safetensors
from torch import nn
from torchaudio.models.decoder import ctc_decoder
from torchinfo import summary

import wandb
from architecture import EMGTransformer
from architecture_gaddy import GaddyModel
from data_utils import (
    apply_cli_overrides,
    combine_fixed_length,
    decollate_tensor,
    get_writer,
    load_config,
    make_torch_generator,
    seed_everything,
    seed_worker,
)
from hdf5_dataset import H5EmgDataset, SizeAwareSampler

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
FLAGS = apply_cli_overrides(load_config(os.path.join("config", "recognition.yaml")))
writer = get_writer(FLAGS.log_directory, run_id)


def build_model(num_features: int, num_outs: int, model_name: str | None = None):
    """Build the configured recognition architecture."""
    model_name = model_name or FLAGS.model
    if model_name == "gaddy":
        return GaddyModel(
            num_features=num_features,
            num_outs=num_outs,
            model_size=FLAGS.gaddy_model_size,
            num_layers=FLAGS.gaddy_num_layers,
            dropout=FLAGS.gaddy_dropout,
        )
    if model_name == "tinymyo":
        return EMGTransformer(
            num_features=num_features,
            num_outs=num_outs,
            in_chans=FLAGS.in_chans,
            embed_dim=FLAGS.embed_dim,
            n_layer=FLAGS.num_layers,
            n_head=FLAGS.num_heads,
            mlp_ratio=FLAGS.mlp_ratio,
            attn_drop=FLAGS.dropout,
            proj_drop=FLAGS.dropout,
        )
    raise ValueError(f"Unknown recognition model: {model_name!r}")


def load_starting_state_dict(checkpoint_path: str) -> dict:
    """Load either a native PyTorch/Lightning or a safetensors checkpoint."""
    checkpoint_path = checkpoint_path.strip()
    if checkpoint_path.lower().endswith(".safetensors"):
        state_dict = load_safetensors(checkpoint_path, device="cpu")
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)

    return {
        key.removeprefix("model."): value
        for key, value in state_dict.items()
        if not key.endswith("num_batches_tracked")
    }


def evaluate_wer(
    model: nn.Module,
    dset: H5EmgDataset,
    device: str,
    beam_size: int = 150,
) -> float:
    """
    Evaluate WER using the KenLM + Flashlight CTC beam-search decoder.
    """
    tkns = list(dset.text_transform.chars) + ["_"]
    decoder = ctc_decoder(
        lexicon=os.path.join(
            FLAGS.lm_directory,
            "gaddy_derived_lexicon.txt",
        ),
        tokens=tkns,
        lm=os.path.join(
            FLAGS.lm_directory,
            "lm.bin",
        ),
        lm_dict=None,
        blank_token="_",
        sil_token="|",
        nbest=1,
        lm_weight=4.0,
        word_score=-1.75,
        sil_score=0.0,
        beam_size=beam_size,
    )

    dataloader = torch.utils.data.DataLoader(
        dset,
        batch_size=1,
        pin_memory=(device == "cuda"),
        collate_fn=dset.collate_raw,
        num_workers=FLAGS.num_workers,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=make_torch_generator(int(FLAGS.seed)),
    )

    references = []
    predictions = []

    model.eval()
    with torch.no_grad():
        for example in tqdm.tqdm(dataloader, "WER evaluation"):
            X = example["emg"][0].unsqueeze(0).to(device)
            X_raw = example["raw_emg"][0].unsqueeze(0).to(device)
            sess = example["session_ids"][0].to(device)

            pred = model(X, X_raw, sess)
            pred = F.log_softmax(pred, dim=-1)

            beam_results = decoder(pred.detach().cpu())

            pred_text = " ".join(beam_results[0][0].words).strip()

            pred_text = dset.text_transform.clean_text(pred_text)

            target_text = dset.text_transform.clean_text(example["text"][0])

            if target_text:
                references.append(target_text)
                predictions.append(pred_text)

    model.train()

    if FLAGS.verbose:
        for ref, pred in zip(references, predictions):
            print(f"Ref: {ref}")
            print(f"Pred: {pred}")
            print("---" * 20)

    return jiwer.wer(
        references,
        predictions,
    )


def evaluate_ctc_loss(
    model: nn.Module,
    dset: H5EmgDataset,
    device: str,
) -> float:
    """
    Compute CTC loss on a dataset without beam decoding.
    """
    model.eval()

    dataloader = torch.utils.data.DataLoader(
        dset,
        pin_memory=(device == "cuda"),
        num_workers=FLAGS.num_workers,
        collate_fn=dset.collate_raw,
        batch_sampler=SizeAwareSampler(
            dset,
            128_000,
            seed=int(FLAGS.seed),
        ),
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=make_torch_generator(int(FLAGS.seed)),
    )

    n_chars = len(dset.text_transform.chars)
    losses = []

    with torch.no_grad():
        for example in tqdm.tqdm(
            dataloader,
            "Validation",
            disable=None,
        ):
            X = combine_fixed_length(
                example["emg"],
                FLAGS.seq_len,
            ).to(device)

            X_raw = combine_fixed_length(
                example["raw_emg"],
                FLAGS.seq_len * 8,
            ).to(device)

            sess = combine_fixed_length(
                example["session_ids"],
                FLAGS.seq_len,
            ).to(device)

            pred = model(
                X,
                X_raw,
                sess,
            )

            pred = F.log_softmax(
                pred,
                dim=-1,
            )

            pred = nn.utils.rnn.pad_sequence(
                decollate_tensor(
                    pred,
                    example["lengths"],
                ),
                batch_first=False,
            )

            y = nn.utils.rnn.pad_sequence(
                example["text_int"],
                batch_first=True,
            ).to(device)

            loss = F.ctc_loss(
                pred,
                y,
                example["lengths"],
                example["text_int_lengths"],
                blank=n_chars,
            )

            losses.append(loss.item())

    model.train()

    return float(np.mean(losses))


def train_model(
    model: nn.Module,
    trainset: H5EmgDataset,
    devset: H5EmgDataset,
    device: str,
) -> nn.Module:

    dataloader = torch.utils.data.DataLoader(
        trainset,
        pin_memory=(device == "cuda"),
        num_workers=FLAGS.num_workers,
        collate_fn=trainset.collate_raw,
        batch_sampler=SizeAwareSampler(
            trainset,
            128_000,
            seed=int(FLAGS.seed),
        ),
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=make_torch_generator(int(FLAGS.seed)),
    )

    n_chars = len(devset.text_transform.chars)

    if FLAGS.start_training_from is not None:
        state_dict = load_starting_state_dict(FLAGS.start_training_from)

        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict,
            strict=False,
        )

        print(f"Missing keys when loading model: {missing_keys}")

        print(f"Unexpected keys when loading model: {unexpected_keys}")

        logging.info(f"Loaded model from {FLAGS.start_training_from}")

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=FLAGS.learning_rate,
        weight_decay=FLAGS.weight_decay,
    )

    lr_sched = torch.optim.lr_scheduler.MultiStepLR(
        optim,
        milestones=list(FLAGS.learning_rate_milestones),
        gamma=FLAGS.learning_rate_gamma,
    )

    def set_lr(new_lr):
        for param_group in optim.param_groups:
            param_group["lr"] = new_lr

    target_lr = FLAGS.learning_rate

    def schedule_lr(iteration):
        iteration += 1

        if iteration <= FLAGS.learning_rate_warmup:
            set_lr(iteration * target_lr / FLAGS.learning_rate_warmup)

    batch_idx = 0

    run_checkpoint_directory = os.path.join(
        FLAGS.ckpt_directory,
        FLAGS.model,
        run_id,
    )
    os.makedirs(run_checkpoint_directory, exist_ok=True)
    logging.info(f"Run checkpoints: {run_checkpoint_directory}")

    if FLAGS.wandb_logging:
        wandb.init(
            project=FLAGS.task,
            config=FLAGS,
            name=f"{FLAGS.task}_{run_id}",
        )

    # Important because gradient accumulation is used.
    optim.zero_grad(set_to_none=True)

    for epoch_idx in range(FLAGS.num_epochs):
        model.train()

        train_losses = []

        for example in tqdm.tqdm(
            dataloader,
            "Train step",
            disable=None,
        ):
            schedule_lr(batch_idx)

            X = combine_fixed_length(
                example["emg"],
                FLAGS.seq_len,
            ).to(device)

            X_raw = combine_fixed_length(
                example["raw_emg"],
                FLAGS.seq_len * 8,
            ).to(device)

            sess = combine_fixed_length(
                example["session_ids"],
                FLAGS.seq_len,
            ).to(device)

            pred = model(
                X,
                X_raw,
                sess,
            )

            pred = F.log_softmax(
                pred,
                dim=-1,
            )

            pred = nn.utils.rnn.pad_sequence(
                decollate_tensor(
                    pred,
                    example["lengths"],
                ),
                batch_first=False,
            )

            y = nn.utils.rnn.pad_sequence(
                example["text_int"],
                batch_first=True,
            ).to(device)

            loss = F.ctc_loss(
                pred,
                y,
                example["lengths"],
                example["text_int_lengths"],
                blank=n_chars,
            )

            train_losses.append(loss.item())

            writer.add_scalar(
                "train/loss_step",
                loss.item(),
                batch_idx,
            )

            if FLAGS.wandb_logging:
                wandb.log(
                    {"train/loss_step": loss.item()},
                )

            loss.backward()

            if (batch_idx + 1) % 2 == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                )

                optim.step()

                optim.zero_grad(set_to_none=True)

            batch_idx += 1

        train_loss = float(np.mean(train_losses))

        val_loss = evaluate_ctc_loss(
            model,
            devset,
            device,
        )

        logging.info(
            f"finished epoch {epoch_idx + 1} - "
            f"training loss: {train_loss:.4f} "
            f"validation loss: {val_loss:.4f}"
        )

        wer_interval = max(1, int(getattr(FLAGS, "eval_interval", 5)))
        if (epoch_idx + 1) % wer_interval == 0 or epoch_idx == FLAGS.num_epochs - 1:
            dev_wer = evaluate_wer(
                model=model,
                dset=devset,
                device=device,
                beam_size=150,
            )
            logging.info(
                f"finished epoch {epoch_idx + 1} - dev WER: {dev_wer * 100:.2f}%"
            )
            writer.add_scalar("val/wer", dev_wer, epoch_idx)
            if FLAGS.wandb_logging:
                wandb.log({"val/wer": dev_wer, "epoch": epoch_idx + 1})

        lr_sched.step()
        current_lr = optim.param_groups[0]["lr"]

        checkpoint_path = os.path.join(
            run_checkpoint_directory,
            f"epoch={epoch_idx + 1:03d}.pt",
        )
        torch.save(model.state_dict(), checkpoint_path)
        logging.info(f"Saved checkpoint: {checkpoint_path}")

        writer.add_scalar(
            "train/loss_epoch",
            train_loss,
            epoch_idx,
        )

        writer.add_scalar(
            "val/loss",
            val_loss,
            epoch_idx,
        )

        writer.add_scalar(
            "train/lr",
            current_lr,
            epoch_idx,
        )

        if FLAGS.wandb_logging:
            wandb.log(
                {
                    "train/loss_epoch": train_loss,
                    "val/loss": val_loss,
                    "lr": current_lr,
                    "epoch": epoch_idx + 1,
                }
            )

    return model


def evaluate_saved():
    seed_everything(int(FLAGS.seed))
    device = "cuda" if torch.cuda.is_available() and not FLAGS.debug else "cpu"

    dev = FLAGS.dev

    testset = H5EmgDataset(
        dev=dev,
        test=not dev,
        config=FLAGS,
    )

    silent_flags = [d.silent for (d, _) in testset.example_indices]

    print(f"Unique silent flags in test set: {set(silent_flags)}")

    n_chars = len(testset.text_transform.chars)

    model = build_model(testset.num_features, n_chars + 1).to(device)

    model.load_state_dict(load_starting_state_dict(FLAGS.evaluate_saved), strict=True)

    print(f"Loaded model from {FLAGS.evaluate_saved}")

    wer = evaluate_wer(
        model,
        testset,
        device,
        beam_size=1500,
    )

    print(f"WER: {wer * 100:.2f}%")


def main():
    seed_everything(int(FLAGS.seed))
    os.makedirs(FLAGS.log_directory, exist_ok=True)
    os.makedirs(FLAGS.output_directory, exist_ok=True)
    os.makedirs(FLAGS.ckpt_directory, exist_ok=True)
    logging.basicConfig(
        handlers=[
            logging.FileHandler(
                os.path.join(FLAGS.log_directory, f"train_{FLAGS.task}_{run_id}.log")
            ),
            logging.StreamHandler(),
        ],
        level=logging.INFO,
        format="%(message)s",
    )

    logging.info(sys.argv)

    trainset = H5EmgDataset(dev=False, test=False, config=FLAGS)
    devset = H5EmgDataset(dev=True, config=FLAGS)
    testset = H5EmgDataset(test=True, config=FLAGS)
    logging.info("output example: %s", devset.example_indices[0])
    logging.info("train / dev split: %d %d", len(trainset), len(devset))

    device = "cuda" if torch.cuda.is_available() and not FLAGS.debug else "cpu"

    n_chars = len(devset.text_transform.chars)
    model = build_model(devset.num_features, n_chars + 1).to(device)
    summary(
        model,
        input_data=[
            torch.randn(1, FLAGS.full_seq_len, FLAGS.in_chans).to(device),
            torch.randn(1, FLAGS.full_seq_len, FLAGS.in_chans).to(device),
            torch.randn(1, FLAGS.full_seq_len, FLAGS.in_chans).to(device),
        ],
    )

    # FLOPs
    flops = torchprofile.profile_macs(
        model,
        (
            torch.randn(1, FLAGS.full_seq_len, FLAGS.in_chans).to(device),
            torch.randn(1, FLAGS.full_seq_len, FLAGS.in_chans).to(device),
            torch.randn(1, FLAGS.full_seq_len, FLAGS.in_chans).to(device),
        ),
    )
    logging.info(f"FLOPs: {flops / 1e9:.4f} G")

    best_model = train_model(
        model,
        trainset,
        devset,
        device,
    )

    # Run test
    test_wer = evaluate_wer(
        model=best_model, dset=testset, device=device, beam_size=1500
    )
    logging.info(
        "Final test WER: %.2f%%",
        test_wer * 100,
    )
    writer.add_scalar("test/wer", test_wer, 0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train or evaluate the recognition model."
    )
    parser.add_argument(
        "--evaluate_saved",
        type=str,
        default=None,
        help="Path to a saved model checkpoint to evaluate on the test set.",
    )
    parser.add_argument(
        "--start_training_from",
        type=str,
        default=None,
        help="Path to a checkpoint or safetensors file used to initialize training.",
    )
    parser.add_argument(
        "--model",
        choices=("tinymyo", "gaddy"),
        default=None,
        required=True,
        help="Recognition architecture to train/evaluate (overrides the config).",
    )
    args, _ = parser.parse_known_args()
    if args.model is not None:
        FLAGS.model = args.model
    if args.evaluate_saved is not None:
        FLAGS.evaluate_saved = args.evaluate_saved
        evaluate_saved()
    else:
        if args.start_training_from is not None:
            FLAGS.start_training_from = args.start_training_from
        main()
