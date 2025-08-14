import os
import logging

import torch
import torchaudio
from speechbrain.inference.ASR import EncoderASR
import jiwer
from unidecode import unidecode
import tqdm

asr = EncoderASR.from_hparams(
    source="speechbrain/asr-wav2vec2-librispeech",
    savedir="pretrained_models/asr-wav2vec2-librispeech",
    run_opts={"device": "cuda"},
)


def evaluate(testset, audio_directory):
    predictions = []
    targets = []

    for i, datapoint in enumerate(tqdm.tqdm(testset, "Evaluate outputs", disable=None)):
        text = asr.transcribe_file(os.path.join(audio_directory, f"example_output_{i}.wav"))

        pred_text = testset.text_transform.clean_text(text)
        target_text = testset.text_transform.clean_text(datapoint['text'])

        predictions.append(pred_text)
        targets.append(target_text)

    for i, (targ, pred) in enumerate(zip(targets, predictions)):
        if targ == '':
            del targets[i]
            del predictions[i]

    for i in range(len(targets)):
        logging.info(f"Target: {targets[i]}")
        logging.info(f"Prediction: {predictions[i]}")
        logging.info(f"---" * 50)
    logging.info(f"wer: {jiwer.wer(targets, predictions)}")
