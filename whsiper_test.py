import numpy as np
import soundfile as sf
from pydub import AudioSegment
from scipy.signal import resample
from faster_whisper import WhisperModel


data = [1900288, 1525504, 198144, 4260096, 1060864, -1120768, 2136320, 1668864, 1662720, 
        188160, 4329984, 1129472, -919808, 2032896, 1168640, 1970944, 251648, 4598528, 
        1103360, -772608, 2159104, 930304, 2210816, 348928, 4488192, 1162496, -773376, 
        1989632, 763392, 2262016, 463104, 4555008, 1254912, -794880, 2028544, 643840, 
        2422528, 569088, 4628480, 1222144, -631552, 1893376, 539392]

def convert_to_wav(data):
    audio_data = np.array(data, dtype=np.int32)
    # Find the maximum absolute value for normalization
    max_val = np.max(np.abs(audio_data))

    # Normalize data to fit within the 32-bit signed integer range
    normalized_data = (audio_data / max_val * (2**31 - 1)).astype(np.int32)

    # Set parameters for the WAV file
    sample_rate = 16000  # Typical sample rate for speech recognition
    n_channels = 1       # Mono audio
    sampwidth = 4        # 4 bytes per sample for 32-bit audio

    # Create the WAV file
    output_filename = 'output_32bit.wav'
    with wave.open(output_filename, 'w') as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(normalized_data.tobytes())

    print(f"WAV file created: {output_filename}")


def load_mp3_as_numpy(file_path, target_sample_rate=16000):
    # Load MP3 file
    audio = AudioSegment.from_file(file_path, format="mp3")
    # Check current sample rate
    current_sample_rate = audio.frame_rate

    # Convert to a NumPy array
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

    # Reshape for stereo audio
    if audio.channels == 2:
        samples = samples.reshape((-1, 2))
        samples = samples.mean(axis=1)  # Convert to mono by averaging channels

    # Resample if needed
    if current_sample_rate != target_sample_rate:
        num_samples = int(len(samples) * float(target_sample_rate) / current_sample_rate)
        samples = resample(samples, num_samples)

    # Normalize to range -1 to 1
    samples = samples / np.max(np.abs(samples))

    return samples, target_sample_rate


def transcribe_audio(model, audio_data, sample_rate=16000):
    # Transcribe using Whisper
    segments, info = model.transcribe(audio_data, beam_size=5, language='en')
    return segments, info


def print_transcription_output(segments):
    for segment in segments:
        print(f"Start: {segment.start:.2f} seconds")
        print(f"End: {segment.end:.2f} seconds")
        print(f"Text: {segment.text}")
        print(f"Tokens: {segment.tokens}")


# Path to the MP3 file
mp3_path = "common_voice_en_39586346.mp3"

# Load and preprocess audio
# audio_data, sample_rate = load_mp3_as_numpy(mp3_path)
# audio_data = np.array(data, dtype=np.int32)
import wave
convert_to_wav(data)

# Initialize the model
# model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

# Perform transcription
# segments, info = transcribe_audio(model, audio_data)

# Print the transcription output
# print_transcription_output(segments)
