from faster_whisper import WhisperModel
import numpy as np
import io
import wave
import struct

# Initialize the Whisper model
model_size = "large-v3"
model = WhisperModel(model_size, device="cpu", compute_type="int8")

# Raw audio data (replace with your actual data)
data = [-709632, -919040, 1998592, -965888, -1802240, 121088, -562176, -644864, -914432, 2013952, -884736, -1742848, 141312, -605184, -628736, -927744, 2023680, -874496, -1564672, 317952, -645120, -714496, -1084416, 1977344, -870144, -1463808, 104704, -607232, -651264, -972544, 2007808, -918528, -1560576, 207616, -661760, -670208, -980992, 1940992, -1013504, -1549056, 335360, -625664, -710400, -1006080, 2012160, -982272, -1497088, 76032, -599552, -734208, -972800, 2007808, -819200, -1518080, 148224, -520448, -692736, -865536, 2119168, -942848, -1686016, 161792, -708608, -708608, -1023232, 2035712, -973824, -1554432, 156416, -690176, -673792, -948224, 2006528, -1120000, -1604608, 281088, -679936, -651264, -927232, 2012416, -1087232, -1486848, 59648, -582656, -706304, -1002496, 1964544, -976896, -1474816, 304896, -643584, -715520, -1093888, 2072064, -1005824, -1665024, 192768, -584192, -705280, -1252352, 1942016, -1021952, -1423360, 78592, -660224, -679424, -1050624, 1947136, -958208, -1657088, 104192, -654592, -641280, -1032448, 1971712, -987392, -1662208, 111872, -529408, -687872, -1127168, 2030336, -1006080, -1585920, 48128, -640512, -658176, -1118720, 1949696, -957952, -1634560, 114176, -733184, -774144, -1170688, 1977600, -1044736]

# Normalize the data to fit within 16-bit PCM range
data = np.array(data)
min_val, max_val = data.min(), data.max()
data = np.interp(data, (min_val, max_val), (-32768, 32767)).astype(np.int16)

# Convert to byte data
byte_data = struct.pack('<' + 'h' * len(data), *data)

# Create an in-memory file-like object and write the WAV header and data
audio_io = io.BytesIO()
with wave.open(audio_io, 'wb') as wav_file:
    wav_file.setnchannels(1)         # Mono audio
    wav_file.setsampwidth(2)         # 2 bytes for 16-bit audio
    wav_file.setframerate(16000)     # 16 kHz sample rate
    wav_file.writeframes(byte_data)  # Write the PCM byte data

# Reset the file pointer to the beginning
audio_io.seek(0)

# Transcribe the audio data
try:
    segments, info = model.transcribe(audio_io, beam_size=5)

    print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")

    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
except Exception as e:
    print("An error occurred during transcription:", e)
