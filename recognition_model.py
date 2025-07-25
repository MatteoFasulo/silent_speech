import os
import sys
from datetime import datetime
import numpy as np
import logging
import multiprocessing
from pyctcdecode import build_ctcdecoder
import jiwer
import tqdm

from timm.scheduler.cosine_lr import CosineLRScheduler
import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

from hdf5_dataset import H5EmgDataset, SizeAwareSampler
#from architecture import Model
from models import Conformer
from data_utils import combine_fixed_length, decollate_tensor
from get_vocab import UNIGRAMS

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('debug', False, 'debug')
flags.DEFINE_string('output_directory', 'output', 'where to save models and outputs')
flags.DEFINE_integer('num_epochs', 200, 'number of epochs')
flags.DEFINE_float('learning_rate', 3e-4, 'learning rate')
flags.DEFINE_integer('learning_rate_warmup', 1000, 'steps of linear warmup')
flags.DEFINE_float('l2', 0, 'weight decay')
flags.DEFINE_integer('gradient_accumulation_steps', 2, 'gradient accumulation steps')
flags.DEFINE_integer('evaluation_frequency', 5, 'how often to evaluate on dev set')
flags.DEFINE_string('start_training_from', None, 'start training from this model')
flags.DEFINE_float('data_size_fraction', 1.0, 'fraction of training data to use')
flags.DEFINE_string('evaluate_saved', None, 'run evaluation on given model file')
flags.DEFINE_integer('num_workers', 64, 'number of workers for dataloaders')
flags.DEFINE_integer('beam_width', 250, 'beam width for CTC decoder')

run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

def get_ctc_decoder(vocab, kenlm_model_path='lm.binary', unigrams=UNIGRAMS, alpha=0.5, beta=1.5):
    return build_ctcdecoder(
        labels=vocab,
        kenlm_model_path=kenlm_model_path,
        unigrams=unigrams,
        alpha=alpha,
        beta=beta,
    )

@torch.no_grad()
def test(model, dset, device):
    model.eval()

    tkns = [c for c in dset.text_transform.chars] + ['_']
    decoder = get_ctc_decoder(vocab=tkns)

    dataloader = torch.utils.data.DataLoader(
        dset, 
        batch_size=1, 
        pin_memory=(device=='cuda'),
        num_workers=FLAGS.num_workers,
        persistent_workers=True,
    )

    logits_list = []
    references = []

    for example in tqdm.tqdm(dataloader, 'Evaluate', disable=None):
        X_raw = example['raw_emg'].to(device)
        lengths = torch.tensor([X_raw.size(1)], dtype=torch.long, device=device) // FLAGS.downsample_factor

        logits, out_lengths = model(X_raw, lengths)
        pred_probs = F.log_softmax(logits, dim=-1)

        # trim to the valid frames only
        T_valid = out_lengths[0].item()
        valid_probs = pred_probs[0, :T_valid, :]

        logits_list.append(valid_probs.cpu().numpy())
        references.append(dset.text_transform.clean_text(example['text'][0]))

    with multiprocessing.get_context("fork").Pool() as pool:
        predictions = decoder.decode_batch(
            pool, 
            logits_list,
            beam_width=FLAGS.beam_width
        )

    for ref, pred in zip(references, predictions):
        print(f"Reference: {ref}")
        print(f"Prediction: {pred}")
        print("---")

    return jiwer.wer(references, predictions)

@torch.no_grad()
def compute_wer(
        model, 
        dataset, 
        device, 
        stage: str = 'dev', 
        global_step: int = 0,
    ):
    model.eval()

    tkns = [c for c in dataset.text_transform.chars] + ['_']
    decoder = get_ctc_decoder(vocab=tkns)

    if stage == 'train':
        num_samples = min(len(dataset), 200)
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        subset = torch.utils.data.Subset(dataset, indices)
    else:
        subset = dataset

    dataloader = torch.utils.data.DataLoader(
        subset,
        batch_size=1,
        num_workers=FLAGS.num_workers,
        pin_memory=(device == 'cuda'),
        persistent_workers=True,
        
    )

    logits_list = []
    references = []

    for example in tqdm.tqdm(dataloader, desc=f"{stage} WER", leave=False):
        X_raw = example['raw_emg'].to(device)
        lengths = torch.tensor([X_raw.size(1)], dtype=torch.long, device=device) // FLAGS.downsample_factor

        logits, out_lengths = model(X_raw, lengths)
        pred_probs = F.log_softmax(logits, dim=-1)

        # trim to the valid frames only
        T_valid = out_lengths[0].item()
        valid_probs = pred_probs[0, :T_valid, :]

        logits_list.append(valid_probs.cpu().numpy())
        references.append(dataset.text_transform.clean_text(example['text'][0]))

    with multiprocessing.get_context("fork").Pool() as pool:
        predictions = decoder.decode_batch(
            pool, 
            logits_list,
            beam_width=FLAGS.beam_width
        )

    if stage != 'train' and len(references) > 0:
        writer.add_text(f'{stage}/references', '\n'.join(references[:20]), global_step=global_step)
        writer.add_text(f'{stage}/predictions', '\n'.join(predictions[:20]), global_step=global_step)

    model.train()
    return jiwer.wer(references, predictions)

def train_model(model, trainset, devset, device):

    if FLAGS.data_size_fraction >= 1:
        training_subset = trainset
    else:
        training_subset = trainset.subset(FLAGS.data_size_fraction)

    dataloader = torch.utils.data.DataLoader(
        training_subset, 
        pin_memory=(device=='cuda'), 
        num_workers=FLAGS.num_workers, 
        collate_fn=H5EmgDataset.collate_raw, 
        batch_sampler=SizeAwareSampler(training_subset, max_len=128_000),
        persistent_workers=True
    )

    n_chars = len(devset.text_transform.chars)
    if FLAGS.start_training_from is not None:
        state_dict = torch.load(FLAGS.start_training_from, map_location=torch.device(device))
        model.load_state_dict(state_dict, strict=False)
        logging.info(f"Loaded model from {FLAGS.start_training_from}")

    optim = torch.optim.AdamW(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.l2)

    # Calculate actual updates per epoch for logging
    batches_per_epoch = len(dataloader)
    updates_per_epoch = batches_per_epoch // FLAGS.gradient_accumulation_steps
    total_updates = FLAGS.num_epochs * updates_per_epoch

    scheduler = CosineLRScheduler(
        optim,
        t_initial=total_updates,
        lr_min=7.5e-5,
        warmup_t=FLAGS.learning_rate_warmup,
        warmup_lr_init=1e-6,
        warmup_prefix=True,
        t_in_epochs=False,
    )

    batch_idx = 0
    update_count = 0
    optim.zero_grad()

    logging.info(f'Batches per epoch: {batches_per_epoch}, Updates per epoch: {updates_per_epoch}')
    logging.info(f'Total training updates: {total_updates}')
    logging.info(f'Warmup will last for {FLAGS.learning_rate_warmup / updates_per_epoch:.1f} epochs')
    logging.info(f'Using scheduler: {type(scheduler).__name__}')

    best_wer = float('inf')
    
    for epoch_idx in range(FLAGS.num_epochs):
        losses = []
        for example in tqdm.tqdm(dataloader, 'Train step', disable=None):
            # bucket and pad
            X_raw, raw_lengths = combine_fixed_length(example['raw_emg'], FLAGS.img_size)
            X_raw = X_raw.to(device)
            raw_lengths = raw_lengths.to(device)

            logits, out_lengths = model(X_raw, raw_lengths // FLAGS.downsample_factor)

            log_probs = F.log_softmax(logits, dim=-1)

            # decollate the tensor to match the lengths
            decollated_log_probs = decollate_tensor(log_probs, out_lengths)
            flat_log_probs = torch.cat(decollated_log_probs, dim=0)
            original_out_lengths = [t.shape[0] // FLAGS.downsample_factor for t in example['raw_emg']]
            reconstructed_pred_seqs = torch.split(flat_log_probs, original_out_lengths)
            preds_padded = nn.utils.rnn.pad_sequence(
                reconstructed_pred_seqs,
                batch_first=False,  # CTC requires (Time, Batch, Dim)
            )

            # targets
            targets_padded = nn.utils.rnn.pad_sequence(
                example['text_int'],
                batch_first=True
            ).to(device)
            target_lengths = torch.tensor(example['text_int_lengths'], dtype=torch.long, device=device)

            input_lengths_for_ctc = torch.tensor(original_out_lengths, dtype=torch.long, device=device)

            loss = F.ctc_loss(
                log_probs=preds_padded,
                targets=targets_padded,
                input_lengths=input_lengths_for_ctc,
                target_lengths=target_lengths,
                blank=n_chars,
                zero_infinity=True
            )
            losses.append(loss.item())
            writer.add_scalar('train/loss_step', loss.item(), update_count)

            loss.backward()

            # Update optimizer based on accumulation steps
            if (batch_idx+1) % FLAGS.gradient_accumulation_steps == 0:
                optim.step()
                
                # Step the scheduler
                scheduler.step_update(update_count)
                
                optim.zero_grad()
                update_count += 1

            batch_idx += 1

        train_loss = np.mean(losses)
        current_lr = optim.param_groups[0]['lr']
        writer.add_scalar('train/loss_epoch', train_loss, epoch_idx)
        writer.add_scalar('train/lr', current_lr, epoch_idx)

        if (epoch_idx + 1) % FLAGS.evaluation_frequency == 0:
            train_wer = compute_wer(model, training_subset, device, stage='train')
            val_wer = compute_wer(model, devset, device, stage='dev', global_step=update_count)
            logging.info(
                f"Epoch {epoch_idx}\tLoss {train_loss:.4f}\tTrain WER {train_wer*100:.2f}%\tVal WER {val_wer*100:.2f}%\tLR {current_lr:.2e}"
            )
            writer.add_scalar('val/wer', val_wer, epoch_idx)
            writer.add_scalar('train/wer', train_wer, epoch_idx)
            # Save best model
            if val_wer < best_wer:
                best_wer = val_wer
                ckpt_name = f"model_{run_id}.pt"
                torch.save(model.state_dict(), os.path.join(FLAGS.output_directory, ckpt_name))
                logging.info(f'New best WER: {best_wer*100:.2f} - model saved')
        else:
            logging.info(
                f"Epoch {epoch_idx}\tLoss {train_loss:.4f}\tLR {current_lr:.2e}"
            )

    # Load back the best model
    ckpt_name = f"model_{run_id}.pt"
    best_model_path = os.path.join(FLAGS.output_directory, ckpt_name)
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=torch.device(device)))
        logging.info(f"Best model loaded from {best_model_path}")
        
    return model

def evaluate_saved():
    device = 'cuda' if torch.cuda.is_available() and not FLAGS.debug else 'cpu'
    testset = H5EmgDataset(test=True)
    n_chars = len(testset.text_transform.chars)
    model = Conformer(n_chars+1).to(device)
    model.load_state_dict(torch.load(FLAGS.evaluate_saved, map_location=torch.device(device)))
    summary(
        model, 
        input_data=[
            torch.randn(1, FLAGS.img_size, FLAGS.in_chans).to(device),
            torch.tensor([FLAGS.img_size], dtype=torch.long, device=device) // FLAGS.downsample_factor
        ],
    )
    print(f"Loaded model from {FLAGS.evaluate_saved}")
    test_wer = test(model, testset, device)
    print('WER:', test_wer)

def main():
    log_filename = os.path.join(FLAGS.output_directory, f'log.txt')
    os.makedirs(FLAGS.output_directory, exist_ok=True)
    logging.basicConfig(handlers=[
            logging.FileHandler(log_filename, 'w'),
            logging.StreamHandler()
            ], level=logging.INFO, format="%(message)s")

    logging.info(sys.argv)

    trainset = H5EmgDataset(dev=False,test=False)
    devset = H5EmgDataset(dev=True)
    testset = H5EmgDataset(test=True)
    logging.info('output example: %s', devset.example_indices[0])
    logging.info('train / dev split: %d %d',len(trainset),len(devset))

    device = 'cuda' if torch.cuda.is_available() and not FLAGS.debug else 'cpu'

    n_chars = len(devset.text_transform.chars)
    model = Conformer(n_chars+1).to(device)
    summary(
        model, 
        input_data=[
            torch.randn(1, FLAGS.img_size, FLAGS.in_chans).to(device),
            torch.tensor([FLAGS.img_size], dtype=torch.long, device=device) // FLAGS.downsample_factor
        ],
    )

    best_ckpt_model = train_model(model, trainset, devset, device)

    # Run test
    test_wer = test(best_ckpt_model, testset, device)
    logging.info('Test WER: %.2f%%', test_wer * 100)
    writer.add_scalar('test/wer', test_wer, 0)
    writer.add_hparams(
        {
            'window_size': FLAGS.img_size,
            'learning_rate': FLAGS.learning_rate,
            'l2': FLAGS.l2,
            'num_epochs': FLAGS.num_epochs,
            'gradient_accumulation_steps': FLAGS.gradient_accumulation_steps,
            'embed_dim': FLAGS.embed_dim,
            'num_heads': FLAGS.num_heads,
            'num_layers': FLAGS.num_layers,
            'downsample_factor': FLAGS.downsample_factor,
            'mlp_ratio': FLAGS.mlp_ratio,
            'beam_width': FLAGS.beam_width,
        },
        {
            'dev_wer': compute_wer(best_ckpt_model, devset, device, stage='dev'),
            'test_wer': test_wer,
        },
    )
    return

if __name__ == '__main__':
    FLAGS(sys.argv)
    if FLAGS.evaluate_saved is not None:
        evaluate_saved()
    else:
        global writer
        writer = SummaryWriter()
        main()
