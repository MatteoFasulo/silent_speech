import os
import logging

import torch
import torchaudio
from speechbrain.inference.ASR import EncoderASR
import jiwer
from unidecode import unidecode
import tqdm

device = "cuda"
asr = EncoderASR.from_hparams(
    source="speechbrain/asr-wav2vec2-librispeech",
    savedir="pretrained_models/asr-wav2vec2-librispeech",
    run_opts={"device": "cuda"},
)


def evaluate(testset, audio_directory):
    predictions = []
    targets = []

    for i, datapoint in enumerate(tqdm.tqdm(testset, "Evaluate outputs", disable=None)):
        audio, rate = torchaudio.load(os.path.join(audio_directory, f"example_output_{i}.wav"))

        if rate != 16000:
            audio = torchaudio.functional.resample(audio, rate, 16000)

        text = asr.transcribe_batch(wavs=audio, wav_lens=torch.tensor([1.0], device=device))

        pred_text = text[0] if isinstance(text[0], str) else str(text[0])

        pred_text = testset.text_transform.clean_text(pred_text)
        target_text = testset.text_transform.clean_text(datapoint["text"])

        predictions.append(pred_text)
        targets.append(target_text)

    for i in range(len(targets)):
        logging.info(f"Target: {targets[i]}")
        logging.info(f"Prediction: {predictions[i]}")
        logging.info(f"---" * 50)
    logging.info(f"wer: {jiwer.wer(targets, predictions)}")
