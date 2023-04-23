from scipy.signal import stft, istft
import tempfile
import numpy as np
import soundfile as sf
import subprocess
import os


def phase_vocoder(audio_data, sample_rate, speed_factor):
    # Parameters for STFT (Short-Time Fourier Transform)
    window_size = 1024
    hop_length = int(window_size // 4)

    # Compute the STFT of the input audio
    f, t, Zxx = stft(audio_data, fs=sample_rate, window='hann', nperseg=window_size, noverlap=window_size - hop_length)

    # Modify the phase information
    num_frames = Zxx.shape[1]
    new_num_frames = int(num_frames / speed_factor)
    phase = np.angle(Zxx)

    new_phase = np.zeros((Zxx.shape[0], new_num_frames), dtype=np.complex64)
    new_magnitude = np.zeros((Zxx.shape[0], new_num_frames), dtype=np.float32)

    for frame_idx in range(new_num_frames):
        new_frame_idx = int(frame_idx * speed_factor)
        if new_frame_idx < num_frames:
            new_phase[:, frame_idx] = phase[:, new_frame_idx]
            new_magnitude[:, frame_idx] = np.abs(Zxx[:, new_frame_idx])

    # Reconstruct the modified STFT using the new phase and magnitude information
    new_Zxx = new_magnitude * np.exp(1j * new_phase)

    # Compute the inverse STFT to obtain the output audio
    _, output_data = istft(new_Zxx, fs=sample_rate, window='hann', nperseg=window_size, noverlap=window_size - hop_length)

    return output_data


def change_speed_without_pitch(audio_data, sample_rate, speed_factor):
    output_data = phase_vocoder(audio_data, sample_rate, speed_factor)
    return output_data


def change_speed_without_pitch_sox(audio_data, sample_rate, speed_factor):
    sox_path = "C:\\Program Files (x86)\\sox-14-4-2\\sox.exe"

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_input_file:
        sf.write(temp_input_file.name, audio_data, samplerate=sample_rate)
        temp_input_file.close()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output_file:
            temp_output_file.close()

            # Change the speed without changing the pitch using the `sox` command
            subprocess.call([sox_path, temp_input_file.name, temp_output_file.name, "tempo", str(speed_factor)])

            # Read the output file
            output_data, _ = sf.read(temp_output_file.name)

            # Remove temporary files
            os.unlink(temp_input_file.name)
            os.unlink(temp_output_file.name)

    return output_data
