from pathlib import Path

import MeCab
import whisper
from jiwer import cer, wer
from s2t_metrics.openai import transcribe as transcribe_whisper_api
from s2t_metrics.google_cloud import transcribe as transcribe_google_api


def transcribe_dummy(model: str, audio_file: str) -> str:
    return 'Dummy'


def transcribe_whisper_cpp(model: str, audio_file: str) -> str:
    return


def transcribe_whisper(model: str, audio_file: str) -> str:
    model = whisper.load_model(model)
    return model.transcribe(audio_file)['text']


def main(input_sound: str, gt_text_file: str):
    mecab = MeCab.Tagger('-Owakati')

    gt_text = Path(gt_text_file).read_text()
    gt_text = mecab.parse(gt_text)
    print(f'GT:{gt_text}')
    # hypothesis = transcribe_whisper('tiny', input_sound)
    # hypothesis = transcribe_whisper_api(input_sound)
    hypothesis = transcribe_google_api(input_sound)
    hypothesis = mecab.parse(hypothesis)
    print(f'hypo:{hypothesis}')
    print(f'WER:{wer(gt_text, hypothesis):.1%}')
    print(f'CER:{cer(gt_text, hypothesis):.1%}')


if __name__ == '__main__':
    main('data/test_1.wav', 'data/test_1_gt.txt')
    main('data/test_2.wav', 'data/test_2_gt.txt')
