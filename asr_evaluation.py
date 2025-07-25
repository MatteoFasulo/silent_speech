import os
import logging

import torch
import torchaudio
from speechbrain.inference.ASR import EncoderASR
import jiwer
from unidecode import unidecode
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
asr = EncoderASR.from_hparams(
    source="speechbrain/asr-wav2vec2-librispeech", run_opts={"device":device}
)

def evaluate(testset, audio_directory):
    predictions = []
    targets = []
    for i, datapoint in enumerate(tqdm.tqdm(testset, 'Evaluate outputs', disable=None)):
        audio, rate = torchaudio.load(os.path.join(audio_directory, f'example_output_{i}.wav'))
        if rate != 16000:
            audio = torchaudio.functional.resample(audio, rate, 16000)
        
        text = asr.transcribe_batch(wavs=audio, wav_lens=torch.tensor([1.0]))
        pred_text = text[0] if isinstance(text[0], str) else str(text[0])
        predictions.append(pred_text)
        target_text = unidecode(datapoint['text'])
        targets.append(target_text)

    transformation = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase()])
    targets = transformation(targets)
    predictions = transformation(predictions)
    logging.info(f'targets: {targets}')
    logging.info(f'predictions: {predictions}')
    logging.info(f'wer: {jiwer.wer(targets, predictions)}')
