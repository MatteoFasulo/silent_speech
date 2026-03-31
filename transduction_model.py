import argparse
import logging
import os
import sys
from datetime import datetime
from typing import List

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from numba import jit
import jiwer

import torch
import torch.nn.functional as F
import torchprofile
import tqdm
from torchinfo import summary
import wandb
from speechbrain.inference.ASR import EncoderASR

from architecture import EMGTransformer
from data_utils import (
    combine_fixed_length,
    decollate_tensor,
    get_writer,
    load_config,
    phoneme_inventory,
    print_confusion,
)
from hdf5_dataset import H5EmgDataset, SizeAwareSampler
from vocoder import Vocoder

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
FLAGS = load_config(os.path.join("config", "transduction_model.json"))
writer = get_writer(FLAGS.log_directory, run_id)


def evaluate(testset: H5EmgDataset, audio_directory: str) -> None:
    """
    Evaluates the model by transcribing generated audio using a pre-trained ASR model
    and calculating the Word Error Rate (WER).

    Args:
        testset (H5EmgDataset): The dataset to evaluate on.
        audio_directory (str): The directory where the generated audio files are stored.
    """
    predictions = []
    targets = []

    # Try to find the model in the local HF cache to support offline HPC environments
    model_source = "speechbrain/asr-wav2vec2-librispeech"
    cache_base = os.path.expanduser("~/.cache/huggingface/hub")

    def get_local_path(repo_id):
        repo_dir = os.path.join(cache_base, f"models--{repo_id.replace('/', '--')}")
        if os.path.exists(repo_dir):
            snapshot_dir = os.path.join(repo_dir, "snapshots")
            if os.path.exists(snapshot_dir):
                snapshots = os.listdir(snapshot_dir)
                if snapshots:
                    return os.path.join(snapshot_dir, snapshots[0])
        return repo_id

    model_source = get_local_path(model_source)

    asr = EncoderASR.from_hparams(
        source=model_source,
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        overrides={"wav2vec2": {"source": get_local_path("facebook/wav2vec2-large-960h-lv60-self")}}
    )
    if asr:
        for i, datapoint in enumerate(tqdm.tqdm(testset, "Evaluate outputs", disable=None)):
            text = asr.transcribe_file(os.path.join(audio_directory, f"example_output_{i}.wav"))

            pred_text = testset.text_transform.clean_text(text)
            target_text = testset.text_transform.clean_text(datapoint["text"])

            predictions.append(pred_text)
            targets.append(target_text)

        for i, (targ, _) in enumerate(zip(targets, predictions)):
            if targ == "":
                del targets[i]
                del predictions[i]

        for i in range(len(targets)):
            logging.debug(f"Target: {targets[i]}")
            logging.debug(f"Prediction: {predictions[i]}")
            logging.debug("---" * 50)
        logging.info(f"WER: {jiwer.wer(targets, predictions)}")

@jit(nopython=True)
def time_warp(costs: np.ndarray) -> np.ndarray:
    """
    Computes the Dynamic Time Warping (DTW) cost matrix for the given distance matrix.

    Args:
        costs: A 2D array of shape (seq1_len, seq2_len) containing pairwise distances.

    Returns:
        dtw: A 2D array of the same shape as costs, where dtw[i, j] is the minimum cumulative cost
    """
    dtw = np.zeros_like(costs)
    dtw[0, 1:] = np.inf
    dtw[1:, 0] = np.inf
    eps = 1e-4
    for i in range(1, costs.shape[0]):
        for j in range(1, costs.shape[1]):
            dtw[i, j] = costs[i, j] + min(
                dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1]
            )
    return dtw


def align_from_distances(distance_matrix: np.ndarray, debug: bool = False) -> List[int]:
    """
    Computes an alignment between two sequences given a distance matrix using Dynamic Time Warping (DTW).

    Args:
        distance_matrix: A 2D array of shape (seq1_len, seq2_len) containing pairwise distances.
        debug: If True, will display a visualization of the alignment.

    Returns:
        A list of indices where the i-th element is the index in the second sequence that best aligns with the i-th element of the first sequence.
    """
    # for each position in spectrum 1, returns best match position in spectrum2
    # using monotonic alignment
    dtw = time_warp(distance_matrix)

    i = distance_matrix.shape[0] - 1
    j = distance_matrix.shape[1] - 1
    results = [0] * distance_matrix.shape[0]
    while i > 0 and j > 0:
        results[i] = j
        i, j = min(
            [(i - 1, j), (i, j - 1), (i - 1, j - 1)], key=lambda x: dtw[x[0], x[1]]
        )

    if debug:
        visual = np.zeros_like(dtw)
        visual[range(len(results)), results] = 1
        plt.matshow(visual)
        plt.show()

    return results


def test(model: torch.nn.Module, testset: H5EmgDataset, device: str) -> tuple[float, float, np.ndarray]:
    """
    Performs validation on the provided test set, calculating loss and phoneme accuracy.

    Args:
        model (torch.nn.Module): The model to evaluate.
        testset (H5EmgDataset): The dataset to use for validation.
        device (str): The device (cpu or cuda) to run evaluation on.

    Returns:
        tuple[float, float, np.ndarray]: A tuple containing mean loss, mean phoneme accuracy,
                                         and the phoneme confusion matrix.
    """
    model.eval()

    dataloader = torch.utils.data.DataLoader(testset, batch_size=32, collate_fn=testset.collate_raw)
    losses = []
    accuracies = []
    phoneme_confusion = np.zeros((len(phoneme_inventory), len(phoneme_inventory)))
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, "Validation", disable=None):
            X = combine_fixed_length([t.to(device, non_blocking=True) for t in batch["emg"]], FLAGS.seq_len)
            X_raw = combine_fixed_length([t.to(device, non_blocking=True) for t in batch["raw_emg"]], FLAGS.seq_len * 8)
            sess = combine_fixed_length([t.to(device, non_blocking=True) for t in batch["session_ids"]], FLAGS.seq_len)

            pred, phoneme_pred = model(X, X_raw, sess)

            loss, phon_acc = dtw_loss(pred, phoneme_pred, batch, True, phoneme_confusion)
            losses.append(loss.item())

            accuracies.append(phon_acc)

    model.train()
    return (
        np.mean(losses),
        np.mean(accuracies),
        phoneme_confusion,
    )


def save_output(
    model: torch.nn.Module,
    datapoint: dict,
    filename: str,
    device: str,
    audio_normalizer: object,
    vocoder: Vocoder,
) -> None:
    """
    Generates audio from a model prediction for a single datapoint and saves it to a file.

    Args:
        model (torch.nn.Module): The model used for inference.
        datapoint (dict): The sample to use for generating audio.
        filename (str): The output filename to save the audio.
        device (str): The device to use for computation.
        audio_normalizer (object): Object used for inverse normalization of MFCC features.
        vocoder (Vocoder): The vocoder used to generate wav files from features.
    """
    model.eval()
    with torch.no_grad():
        sess = datapoint["session_ids"].to(device=device).unsqueeze(0)
        X = datapoint["emg"].to(dtype=torch.float32, device=device).unsqueeze(0)
        X_raw = datapoint["raw_emg"].to(dtype=torch.float32, device=device).unsqueeze(0)

        pred, _ = model(X, X_raw, sess)
        y = pred.squeeze(0)

        y = audio_normalizer.inverse(y.cpu()).to(device)

        audio = vocoder(y).cpu().numpy()

    sf.write(filename, audio, 22050)

    model.train()


def get_aligned_prediction(
    model: torch.nn.Module,
    datapoint: dict,
    device: str,
    audio_normalizer: object,
) -> torch.Tensor:
    """
    Gets model predictions and optionally aligns them with target features using DTW if silent.

    Args:
        model (torch.nn.Module): The model to use.
        datapoint (dict): The data sample.
        device (str): The device to run calculation on.
        audio_normalizer (object): Normalizer for scaling features back.

    Returns:
        torch.Tensor: The predicted (and possibly aligned) audio features.
    """
    model.eval()
    with torch.no_grad():
        silent = datapoint["silent"]
        sess = datapoint["session_ids"].to(device).unsqueeze(0)
        X = datapoint["emg"].to(device).unsqueeze(0)
        X_raw = datapoint["raw_emg"].to(device).unsqueeze(0)
        y = datapoint["parallel_voiced_audio_features" if silent else "audio_features"].to(device).unsqueeze(0)

        pred, _ = model(X, X_raw, sess)  # (1, seq, dim)

        if silent:
            costs = torch.cdist(pred, y).squeeze(0)
            alignment = align_from_distances(costs.T.detach().cpu().numpy())
            pred_aligned = pred.squeeze(0)[alignment]
        else:
            pred_aligned = pred.squeeze(0)

        pred_aligned = audio_normalizer.inverse(pred_aligned.cpu())

    model.train()
    return pred_aligned


def dtw_loss(
    predictions: torch.Tensor,
    phoneme_predictions: torch.Tensor,
    example: dict,
    phoneme_eval: bool = False,
    phoneme_confusion: np.ndarray | None = None,
) -> tuple[torch.Tensor, float]:
    """
    Computes a loss between prediction and audio using Dynamic Time Warping (DTW) for silent speech.
    Also calculates phoneme classification accuracy for both silent and voiced speech.

    Args:
        predictions (torch.Tensor): Audio feature predictions from the model.
        phoneme_predictions (torch.Tensor): Phoneme class log-probabilities or logits.
        example (dict): A batch from the dataloader containing ground truth.
        phoneme_eval (bool, optional): Whether to calculate confusion matrix and accuracy.
        phoneme_confusion (np.ndarray, optional): Accumulator of phoneme confusion.

    Returns:
        tuple[torch.Tensor, float]: Mean loss per sequence frame and phoneme accuracy.
    """
    device = predictions.device

    predictions = decollate_tensor(predictions, example["lengths"])
    phoneme_predictions = decollate_tensor(phoneme_predictions, example["lengths"])

    audio_features = [t.to(device, non_blocking=True) for t in example["audio_features"]]

    phoneme_targets = example["phonemes"]

    losses = []
    correct_phones = 0
    total_length = 0
    for pred, y, pred_phone, y_phone, silent in zip(
        predictions,
        audio_features,
        phoneme_predictions,
        phoneme_targets,
        example["silent"],
    ):
        assert len(pred.size()) == 2 and len(y.size()) == 2
        y_phone = y_phone.to(device)

        if silent:
            dists = torch.cdist(pred.unsqueeze(0), y.unsqueeze(0))
            costs = dists.squeeze(0)

            # pred_phone (seq1_len, 48), y_phone (seq2_len)
            # phone_probs (seq1_len, seq2_len)
            pred_phone = F.log_softmax(pred_phone, -1)
            phone_lprobs = pred_phone[:, y_phone]

            costs = costs + FLAGS.phoneme_loss_weight * -phone_lprobs

            alignment = align_from_distances(costs.T.cpu().detach().numpy())

            loss = costs[alignment, range(len(alignment))].sum()

            if phoneme_eval:
                alignment = align_from_distances(costs.T.cpu().detach().numpy())

                pred_phone = pred_phone.argmax(-1)
                correct_phones += (pred_phone[alignment] == y_phone).sum().item()

                for p, t in zip(pred_phone[alignment].tolist(), y_phone.tolist()):
                    phoneme_confusion[p, t] += 1
        else:
            assert y.size(0) == pred.size(0)

            dists = F.pairwise_distance(y, pred)

            assert len(pred_phone.size()) == 2 and len(y_phone.size()) == 1
            phoneme_loss = F.cross_entropy(pred_phone, y_phone, reduction="sum")
            loss = dists.sum() + FLAGS.phoneme_loss_weight * phoneme_loss

            if phoneme_eval:
                pred_phone = pred_phone.argmax(-1)
                correct_phones += (pred_phone == y_phone).sum().item()

                for p, t in zip(pred_phone.tolist(), y_phone.tolist()):
                    phoneme_confusion[p, t] += 1

        losses.append(loss)
        total_length += y.size(0)

    return sum(losses) / total_length, correct_phones / total_length


def train_model(
    trainset: H5EmgDataset,
    devset: H5EmgDataset,
    device: str,
    save_sound_outputs: bool = True,
) -> torch.nn.Module:
    """
    Sets up the model, optimizer, scheduler, and runs the training loop over multiple epochs.

    Args:
        trainset (H5EmgDataset): Dataset for training.
        devset (H5EmgDataset): Dataset for validation.
        device (str): Device to run training on.
        save_sound_outputs (bool): Whether to generate audio samples and evaluate them during training.

    Returns:
        torch.nn.Module: The trained model with best validation loss.
    """
    n_epochs = FLAGS.num_epochs

    if FLAGS.data_size_fraction >= 1:
        training_subset = trainset
    else:
        training_subset = trainset.subset(FLAGS.data_size_fraction)

    dataloader = torch.utils.data.DataLoader(
        training_subset,
        pin_memory=(device == "cuda"),
        collate_fn=devset.collate_raw,
        num_workers=FLAGS.num_workers,
        batch_sampler=SizeAwareSampler(training_subset, 256_000),
        persistent_workers=True,
    )

    n_phones = len(phoneme_inventory)
    model = EMGTransformer(
        num_features=devset.num_features,
        num_outs=devset.num_speech_features,
        num_aux_outs=n_phones,
        in_chans=FLAGS.in_chans,
        embed_dim=FLAGS.embed_dim,
        n_layer=FLAGS.num_layers,
        n_head=FLAGS.num_heads,
        mlp_ratio=FLAGS.mlp_ratio,
        attn_drop=FLAGS.dropout,
        proj_drop=FLAGS.dropout,
        freeze_blocks=FLAGS.freeze_blocks,
    ).to(device)

    # Model summary
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
        args=(
            torch.randn(1, FLAGS.full_seq_len, FLAGS.in_chans).to(device),
            torch.randn(1, FLAGS.full_seq_len, FLAGS.in_chans).to(device),
            torch.randn(1, FLAGS.full_seq_len, FLAGS.in_chans).to(device),
        ),
    )
    logging.info(f"FLOPs: {flops / 1e9:.4f} G")

    if FLAGS.start_training_from is not None:
        state_dict = torch.load(FLAGS.start_training_from, map_location="cpu", weights_only=False)["state_dict"]
        state_dict = {k.replace("model.", "") if k.startswith("model.") else k: v for k, v in state_dict.items()}
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"Missing keys when loading model: {missing_keys}")
        print(f"Unexpected keys when loading model: {unexpected_keys}")
        logging.info(f"Loaded model from {FLAGS.start_training_from}")

    vocoder = None
    if save_sound_outputs:
        vocoder = Vocoder(device=device)

    optim = torch.optim.AdamW(model.parameters(), weight_decay=FLAGS.weight_decay)
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, "min", 0.5, patience=FLAGS.learning_rate_patience)

    def set_lr(new_lr: float):
        for param_group in optim.param_groups:
            param_group["lr"] = new_lr

    target_lr = FLAGS.learning_rate

    def schedule_lr(iteration: int):
        iteration = iteration + 1
        if iteration <= FLAGS.learning_rate_warmup:
            set_lr(iteration * target_lr / FLAGS.learning_rate_warmup)

    batch_idx = 0
    best_val_loss = float("inf")

    if FLAGS.wandb_logging:
        wandb.init(project=FLAGS.wandb_project, config=FLAGS, name=f"{FLAGS.task}_{run_id}", dir=FLAGS.wandb_save_dir)

    for epoch_idx in range(n_epochs):
        losses = []
        for batch in tqdm.tqdm(dataloader, "Train step", disable=None):
            optim.zero_grad()
            schedule_lr(batch_idx)

            X = combine_fixed_length([t.to(device, non_blocking=True) for t in batch["emg"]], FLAGS.seq_len)
            X_raw = combine_fixed_length([t.to(device, non_blocking=True) for t in batch["raw_emg"]], FLAGS.seq_len * 8)
            sess = combine_fixed_length([t.to(device, non_blocking=True) for t in batch["session_ids"]], FLAGS.seq_len)

            pred, phoneme_pred = model(X, X_raw, sess)

            loss, _ = dtw_loss(pred, phoneme_pred, batch)
            losses.append(loss.item())
            writer.add_scalar("train/loss_step", loss.item(), batch_idx)
            if FLAGS.wandb_logging:
                wandb.log({"train/loss_step": loss.item()}, step=batch_idx)

            loss.backward()
            optim.step()

            batch_idx += 1

        train_loss = np.mean(losses)
        val, phoneme_acc, _ = test(model, devset, device)
        lr_sched.step(val)
        current_lr = optim.param_groups[0]["lr"]
        writer.add_scalar("train/loss_epoch", train_loss, epoch_idx)
        writer.add_scalar("train/lr", current_lr, epoch_idx)
        writer.add_scalar("val/loss", val, epoch_idx)
        writer.add_scalar("val/phoneme_acc", phoneme_acc, epoch_idx)
        if FLAGS.wandb_logging:
            wandb.log({
                "train/loss_epoch": train_loss,
                "val/loss": val,
                "val/phoneme_acc": phoneme_acc,
                "lr": current_lr,
                "epoch": epoch_idx
            })
        logging.info(
            f"finished epoch {epoch_idx+1} - validation loss: {val:.4f} training loss: {train_loss:.4f} phoneme accuracy: {phoneme_acc*100:.2f}"
        )

        if val < best_val_loss:
            best_val_loss = val
            torch.save(
                model.state_dict(),
                os.path.join(FLAGS.ckpt_directory, f"model_{run_id}_best.pt"),
            )
            logging.info(f"Val loss improved, new best val loss: {val:.4f}")
        else:
            torch.save(
                model.state_dict(),
                os.path.join(FLAGS.ckpt_directory, f"model_{run_id}_last.pt"),
            )

        if save_sound_outputs:
            save_output(
                model,
                devset[0],
                os.path.join(FLAGS.output_directory, f"epoch_{epoch_idx}_output.wav"),
                device,
                devset.mfcc_norm,
                vocoder,
            )

    if save_sound_outputs:
        for i, datapoint in enumerate(devset):
            save_output(
                model,
                datapoint,
                os.path.join(FLAGS.output_directory, f"example_output_{i}.wav"),
                device,
                devset.mfcc_norm,
                vocoder,
            )

        evaluate(devset, FLAGS.output_directory)

    return model


def main() -> None:
    """
    Main entry point for training the EMG to audio transduction model.
    """
    os.makedirs(FLAGS.log_directory, exist_ok=True)
    os.makedirs(FLAGS.output_directory, exist_ok=True)
    os.makedirs(FLAGS.ckpt_directory, exist_ok=True)
    logging.basicConfig(
        handlers=[
            logging.FileHandler(os.path.join(FLAGS.log_directory, f"train_{FLAGS.task}_{run_id}.log")),
            logging.StreamHandler(),
        ],
        level=logging.INFO,
        format="%(message)s",
    )

    logging.info(sys.argv)

    trainset = H5EmgDataset(dev=False, test=False)
    devset = H5EmgDataset(dev=True)
    logging.info("output example: %s", devset.example_indices[0])
    logging.info("train / dev split: %d %d", len(trainset), len(devset))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_model(
        trainset,
        devset,
        device,
        save_sound_outputs=(FLAGS.hifigan_checkpoint is not None),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate the EMG to audio transduction model.")
    parser.add_argument(
        "--evaluate_saved",
        type=str,
        default=None,
        help="Path to a saved model checkpoint to evaluate on the test set.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output directory for evaluation. Defaults to FLAGS.output_directory.",
    )
    args = parser.parse_args()

    if args.evaluate_saved is not None:
        # Override output directory if provided
        if args.output_dir is not None:
            FLAGS.output_directory = args.output_dir

        os.makedirs(FLAGS.output_directory, exist_ok=True)
        logging.basicConfig(
            handlers=[
                logging.FileHandler(os.path.join(FLAGS.output_directory, "eval_log.txt"), "w"),
                logging.StreamHandler(),
            ],
            level=logging.INFO,
            format="%(message)s",
        )

        testset = H5EmgDataset(dev=FLAGS.dev, test=not FLAGS.dev)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        state_dict = torch.load(args.evaluate_saved, map_location=device, weights_only=False)["state_dict"]
        # Clean state dict if it comes from a PL checkpoint or has "model." prefix
        state_dict = {k.replace("model.", "") if k.startswith("model.") else k: v for k, v in state_dict.items()}

        n_phones = len(phoneme_inventory)
        model = EMGTransformer(
            num_features=testset.num_features,
            num_outs=testset.num_speech_features,
            num_aux_outs=n_phones,
            in_chans=FLAGS.in_chans,
            embed_dim=FLAGS.embed_dim,
            n_layer=FLAGS.num_layers,
            n_head=FLAGS.num_heads,
            mlp_ratio=FLAGS.mlp_ratio,
            attn_drop=FLAGS.dropout,
            proj_drop=FLAGS.dropout,
        ).to(device)
        model.load_state_dict(state_dict, strict=True)

        logging.info(f"Evaluating model from {args.evaluate_saved}")
        _, _, confusion = test(model, testset, device)

        # Save and print confusion matrix
        np.save(os.path.join(FLAGS.output_directory, "confusion_matrix.npy"), confusion)
        print_confusion(confusion)

        vocoder = Vocoder(device=device)
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
    else:
        main()
