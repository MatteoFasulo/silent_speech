import logging
import os

import jiwer
import tqdm
from speechbrain.inference.ASR import EncoderASR


def evaluate(testset, audio_directory):
    predictions = []
    targets = []

    asr = EncoderASR.from_hparams(
        source="speechbrain/asr-wav2vec2-librispeech",
        savedir="pretrained_models/asr-wav2vec2-librispeech",
        run_opts={"device": "cuda"},
    )
    if asr:
        for i, datapoint in enumerate(
            tqdm.tqdm(testset, "Evaluate outputs", disable=None)
        ):
            text = asr.transcribe_file(
                os.path.join(audio_directory, f"example_output_{i}.wav")
            )

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
            logging.debug(f"---" * 50)
        logging.info(f"WER: {jiwer.wer(targets, predictions)}")
