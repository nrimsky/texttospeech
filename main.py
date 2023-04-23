import tempfile
import numpy as np
import soundfile as sf
import subprocess
import os
import torch
from datasets import load_dataset
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from speed import change_speed_without_pitch, change_speed_without_pitch_sox

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def play_audio(data):
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(temp_file.name, data, samplerate=16000)
    temp_file.close()

    # Play the audio using Windows Powershell
    subprocess.call(['powershell.exe', '-c', f'(New-Object Media.SoundPlayer "{temp_file.name}").PlaySync()'])

    # Remove the temporary file
    os.unlink(temp_file.name)


def speak_text(text, model, processor, vocoder, speed_factor=1.5):
    inputs = processor(text=text, return_tensors="pt")
    inputs = inputs.to(device)

    # load xvector containing speaker's voice characteristics from a dataset
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    # Set the temperature for SpeechT5ForTextToSpeech model
    model.config.temperature = 0.01

    speaker_embeddings = speaker_embeddings.to(device)
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

    # Convert tensor to NumPy array
    audio_data = speech.cpu().numpy().astype(np.float32)
    audio_data = audio_data.reshape(-1)

    # Change the speed without changing the pitch

    audio_data_1 = change_speed_without_pitch(audio_data, 16000, speed_factor)

    audio_data_2 = change_speed_without_pitch_sox(audio_data, 16000, speed_factor)

    print("Playing audio...")

    play_audio(audio_data)

    print("(np)")

    play_audio(audio_data_1)

    print("(Sox)")

    play_audio(audio_data_2)


def main():
    print("Device", device)
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    print("loaded processor")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    model = model.to(device)
    print("loaded model")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    vocoder = vocoder.to(device)
    print("loaded vocoder")

    print("Enter 'quit' to exit the program.")
    while True:
        text = input("Enter text to be spoken: ")
        if text.lower() == 'quit':
            break
        speak_text(text, model, processor, vocoder)


if __name__ == "__main__":
    main()

