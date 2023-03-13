import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


def transcribe(audio_file: str) -> str:
    b_file = open(audio_file, "rb")
    transcript = openai.Audio.transcribe("whisper-1", b_file)
    return transcript['text']
