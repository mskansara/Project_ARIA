import argparse
import sys
import aria.sdk as aria
import numpy as np
import wave
import io
import time
from queue import Queue
import threading
import librosa
import webrtcvad

import projectaria_tools
from faster_whisper import WhisperModel
from common import update_iptables
from projectaria_tools.core.sensor_data import (
    ImageDataRecord,
    AudioData,
    AudioDataRecord,
)
# Argument parsing
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interface",
        dest="streaming_interface",
        type=str,
        required=True,
        help="Type of interface to use for streaming. Options are usb or wifi.",
        choices=["usb", "wifi"],
    )
    parser.add_argument(
        "--update_iptables",
        default=False,
        action="store_true",
        help="Update iptables to enable receiving the data stream, only for Linux.",
    )
    parser.add_argument(
        "--profile",
        dest="profile_name",
        type=str,
        default="profile18",
        required=False,
        help="Profile to be used for streaming.",
    )
    parser.add_argument(
        "--device-ip", help="IP address to connect to the device over wifi"
    )
    return parser.parse_args()


# Function to validate raw audio data
def validate_raw_data(raw_data):
    print("Raw data summary:")
    # print(f"Data type: {type(raw_data)}")
    # print(f"Data range: {raw_data.min()} to {raw_data.max()}")
    # print(f"Data shape: {raw_data.shape}")
    # print(f"First few values: {raw_data[:10]}")


# Function to convert raw audio data to WAV format
def convert_to_wav(raw_data, sample_rate=16000):
    # Ensure raw_data is a NumPy array
    raw_data = np.array(raw_data, dtype=np.int32)  # Use int32 to avoid overflow during initial conversion
    # print(type(raw_data))
    # Debugging information
    # validate_raw_data(raw_data)

    # Clamp the values to the valid range of int16
    raw_data = np.clip(raw_data, -32768, 32767)

    # Convert to int16
    audio_data = raw_data.astype(np.int16)

    # Create a BytesIO object to store the WAV data
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16 bits (2 bytes)
        wav_file.setframerate(sample_rate)  # Sample rate
        wav_file.writeframes(audio_data.tobytes())

    # Reset the pointer to the beginning of the BytesIO object
    wav_io.seek(0)
    return wav_io

# Check and normalize the audio data
def check_and_normalize_audio(raw_data):
    # print("Raw data summary:")
    # print(f"Data type: {type(raw_data)}")
    # print(f"Data shape: {raw_data.shape}")
    # print(f"First few raw data values: {raw_data[:10]}")

    # Normalize data to range -1.0 to 1.0
    if raw_data.dtype != np.float32:
        raw_data = raw_data.astype(np.float32)
    audio_data_normalized = np.clip(raw_data / 32768.0, -1.0, 1.0)

    # print("Normalized data sample values:", audio_data_normalized[:10])
    return audio_data_normalized


def check_for_overflows(audio_data):
    # Ensure audio_data is a NumPy array
    if not isinstance(audio_data, np.ndarray):
        audio_data = np.array(audio_data)

    min_val, max_val = -32768, 32767

    # Check for out-of-bound values
    if np.any(audio_data < min_val) or np.any(audio_data > max_val):
        # print("Warning: Audio data contains values outside the int16 range.")
        # print("Values will be clamped to avoid overflow.")
        overflow_values = audio_data[(audio_data < min_val) | (audio_data > max_val)]
        # print(f"Overflow values: {overflow_values}")
        return np.clip(audio_data, min_val, max_val)

    return audio_data

# Constants (updated)
SAMPLE_RATE = 16000
CHUNK_DURATION = 1.0  # Process 1-second chunks for real-time transcription
SCALING_FACTOR = 32767  # Initial scaling factor
FRAME_DURATION_MS = 30  # Frame duration for VAD (10, 20, or 30 ms)
SAMPLES_PER_FRAME = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
# Transcription function
def transcribe_audio(model, audio_queue):
    audio_buffer = []
    while True:
        if not audio_queue.empty():
            audio_data = audio_queue.get()
            audio_data_normalized = librosa.util.normalize(audio_data)
            audio_buffer.extend(audio_data_normalized)
            print(audio_buffer)
            if len(audio_buffer) >= 3 * SAMPLE_RATE:  # Accumulate 3 seconds of speech
                try:
                    segments, info = model.transcribe(np.array(audio_buffer), beam_size=10, language='en')
                    for segment in segments:
                        print("[", segment.start, "-->", segment.end, "]", segment.text)
                except Exception as e:
                    print("Transcription error:", e)

                audio_buffer = []  # Reset buffer
        else:
            time.sleep(0.1)
# Observer class to handle incoming audio data
class StreamingClientObserver:
    def __init__(self, audio_queue):
        self.audio_queue = audio_queue
        self.audio_buffer = []  # Initialize an empty list to buffer audio data
        self.buffer_duration = 20.0  # Desired buffer duration in seconds (e.g., 2 seconds)
        self.sample_rate = 16000  # Sample rate in Hz


    def on_audio_received(self, audio_data: np.array, record:AudioDataRecord):
        # Add received audio data to the buffer
        global SCALING_FACTOR  # Access the global scaling factor variable
        audio_data_clipped = np.clip(audio_data.data, -32768, 32767)
        audio_data_int16 = audio_data_clipped.astype(np.int16)
        max_amplitude = np.max(np.abs(audio_data_int16))

        if max_amplitude > 0:
            SCALING_FACTOR = 32767 / max_amplitude
        self.audio_queue.put(audio_data_int16)


# Main function to set up and manage streaming
def main():
    args = parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()

    aria.set_log_level(aria.Level.Info)
    device_client = aria.DeviceClient()
    client_config = aria.DeviceClientConfig()
    if args.device_ip:
        client_config.ip_v4_address = args.device_ip
    device_client.set_client_config(client_config)
    device = device_client.connect()

    streaming_manager = device.streaming_manager
    streaming_client = streaming_manager.streaming_client

    streaming_config = aria.StreamingConfig()
    streaming_config.profile_name = args.profile_name
    if args.streaming_interface == "usb":
        streaming_config.streaming_interface = aria.StreamingInterface.Usb
    streaming_config.security_options.use_ephemeral_certs = True
    streaming_manager.streaming_config = streaming_config

    streaming_manager.start_streaming()
    streaming_state = streaming_manager.streaming_state
    print(f"Streaming state: {streaming_state}")

    config = streaming_client.subscription_config
    config.subscriber_data_type = aria.StreamingDataType.Audio
    config.message_queue_size[aria.StreamingDataType.Audio] = 10
    options = aria.StreamingSecurityOptions()
    options.use_ephemeral_certs = True
    config.security_options = options
    streaming_client.subscription_config = config

    audio_queue = Queue()
    observer = StreamingClientObserver(audio_queue)
    streaming_client.set_streaming_client_observer(observer)

    print("Start listening to audio data")
    streaming_client.subscribe()

    # print(observer.audio_queue)
    model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

    transcription_thread = threading.Thread(target=transcribe_audio, args=(model, observer.audio_queue), daemon=True)
    transcription_thread.start()
    print("Transcription thread started")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        print("Stop listening to audio data")
        streaming_client.unsubscribe()
        streaming_manager.stop_streaming()
        device_client.disconnect(device)


if __name__ == "__main__":
    main()
