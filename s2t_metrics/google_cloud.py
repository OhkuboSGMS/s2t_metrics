from google.cloud import speech_v1


def transcribe(audio_file: str):
    # Create a client
    client = speech_v1.SpeechClient()

    # Initialize request argument(s)
    config = speech_v1.RecognitionConfig()
    config.language_code = "ja"

    audio = speech_v1.RecognitionAudio()
    with open(audio_file, 'rb') as fp:
        audio.content = fp.read()

    request = speech_v1.RecognizeRequest(
        config=config,
        audio=audio,
    )

    # Make the request
    response = client.recognize(request=request)

    # Handle the response
    print(response)

    return response.results[0].alternatives[0].transcript
