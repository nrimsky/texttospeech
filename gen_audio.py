"""
Script to run the SpeechT5 model for text-to-speech on a book.
Split the book into sentences and generate audio for each sentence.
"""

import numpy as np
import torch
from datasets import load_dataset
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import os
import zipfile


def zip_folder(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, _, files in os.walk(folder_path):
            for file in files:
                zip_file.write(os.path.join(root, file))


def plain_text_to_sentences(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    sentences = text.split('.')
    return sentences

def create_audio(text, model, processor, vocoder, device, max_tokens=400):
    # Function to split text into smaller chunks
    def split_text(text, max_tokens):
        chunks = []
        for i in range(0, len(text), max_tokens):
            chunks.append(text[i:i + max_tokens])
        return chunks

    # Split the input text into smaller chunks
    text_chunks = split_text(text, max_tokens)

    # Initialize an empty list to store audio data
    audio_data_list = []

    # Process each chunk and generate audio
    for chunk in text_chunks:
        inputs = processor(text=chunk, return_tensors="pt")
        inputs = inputs.to(device)

        # Load xvector containing speaker's voice characteristics from a dataset
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

        # Set the temperature for SpeechT5ForTextToSpeech model
        model.config.temperature = 0.01

        speaker_embeddings = speaker_embeddings.to(device)
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

        # Convert tensor to NumPy array
        chunk_audio_data = speech.cpu().numpy().astype(np.float32)
        chunk_audio_data = chunk_audio_data.reshape(-1)

        # Append the audio data to the list
        audio_data_list.append(chunk_audio_data)

    # Concatenate the audio data from all chunks
    audio_data = np.concatenate(audio_data_list)

    return audio_data


def audio_to_mp3(audio_np, filename):
    sf.write(filename, audio_np, 16000, subtype='PCM_16')

def book_convert(filename, path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    model = model.to(device)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    vocoder = vocoder.to(device)
    text = plain_text_to_sentences(filename)
    text = [sentence.strip() for sentence in text]
    text = [sentence+"." for sentence in text if sentence != '']

    # Make directory for audio files
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        # Clear if not empty
        for file in os.listdir(path):
            os.remove(os.path.join(path, file))
    
    for i, sentence in enumerate(text):
        audio_np = create_audio(sentence, model, processor, vocoder, device)
        audio_to_mp3(audio_np, os.path.join(path, f'sen_{i}.wav'))
        print(f"Generated audio for sentence {i+1} of {len(text)}")

    # Save sentences to file
    with open(os.path.join(path, 'sentences.txt'), 'w') as file:
        for sentence in text:
            file.write(sentence + ".")

    # Zip the folder
    zip_folder(path, path + '.zip')

if __name__ == "__main__":
    book_convert('FoundingSales.txt', 'founding_sales_2')