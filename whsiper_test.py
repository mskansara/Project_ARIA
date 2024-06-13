import numpy as np
import soundfile as sf
from pydub import AudioSegment
from scipy.signal import resample
from faster_whisper import WhisperModel


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
audio_data, sample_rate = load_mp3_as_numpy(mp3_path)

# Initialize the model
model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

# Perform transcription
segments, info = transcribe_audio(model, audio_data)

# Print the transcription output
print_transcription_output(segments)
