import argparse
import sys
import aria.sdk as aria
import numpy as np
import wave
import io
import time
from queue import Queue
import threading
from faster_whisper import WhisperModel
from common import update_iptables

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
    print(f"Data type: {type(raw_data)}")
    print(f"Data range: {raw_data.min()} to {raw_data.max()}")
    print(f"Data shape: {raw_data.shape}")
    print(f"First few values: {raw_data[:10]}")


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


# Transcription function
def transcribe_audio(model, audio_queue):
    while True:
        if not audio_queue.empty():
            raw_data = audio_queue.get()
            raw_data = np.array(raw_data, dtype=np.int32)
            # print(type(raw_data))
            try:
                # Validate raw data before conversion
                # validate_raw_data(raw_data)

                audio_io = convert_to_wav(raw_data)
                segments, info = model.transcribe(audio_io, beam_size=5)

                # Validate transcription result
                print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
                transcription = " ".join(segment.text for segment in segments)
                print("Transcription:", transcription.strip())
            except Exception as e:
                print("An error occurred during transcription:", e)
        else:
            time.sleep(0.1)


# Observer class to handle incoming audio data
class StreamingClientObserver:
    def __init__(self, audio_queue):
        self.audio_queue = audio_queue

    def on_audio_received(self, audio_data: np.array, timestamp: int):
        # Debugging information
        # print(f"Received audio data at timestamp: {timestamp}")
        # print(f"Audio data shape: {audio_data.shape}")
        # print(f"First few audio data points: {audio_data[:10]}")

        self.audio_queue.put(audio_data.data)


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

    model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

    transcription_thread = threading.Thread(target=transcribe_audio, args=(model, audio_queue), daemon=True)
    transcription_thread.start()

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
