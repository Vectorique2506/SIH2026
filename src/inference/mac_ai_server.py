"""
Runs on the MacBook. Two jobs:
1. Receives raw mic audio from the Pi -> runs AI model -> sends enhanced audio back to the Pi
2. Separately receives the FINAL NLMS-cleaned audio from the Pi -> plays it out the Mac's own speakers

Requires: pip install onnxruntime numpy librosa sounddevice
"""

import socket
import threading

import numpy as np
import onnxruntime as ort
import librosa
import sounddevice as sd

PI_IP = "169.254.220.174"     # the Pi's confirmed address

AI_LISTEN_PORT = 5005         # Mac listens here for raw mic audio FROM the Pi
AI_SEND_PORT = 5006           # Mac sends AI-enhanced audio back to the Pi on this port
FINAL_LISTEN_PORT = 5007      # Mac listens here for the FINAL NLMS-cleaned audio FROM the Pi

PI_SAMPLE_RATE = 44100
MODEL_SAMPLE_RATE = 16000
CHUNK_SAMPLES_PI_RATE = 4410  # 100ms at 44100Hz

MODEL_PATH = "models_exported/sih_anc_model.onnx"

print("Loading model...")
sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

ai_recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ai_recv_sock.bind(("0.0.0.0", AI_LISTEN_PORT))

ai_send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

final_recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
final_recv_sock.bind(("0.0.0.0", FINAL_LISTEN_PORT))

BYTES_PER_CHUNK = CHUNK_SAMPLES_PI_RATE * 4  # float32 = 4 bytes/sample


def ai_inference_loop():
    """Job 1: mic audio in from Pi -> AI model -> enhanced audio back to Pi."""
    print(f"[AI] Listening on port {AI_LISTEN_PORT}, replying to {PI_IP}:{AI_SEND_PORT}")
    while True:
        data, _ = ai_recv_sock.recvfrom(BYTES_PER_CHUNK + 1024)
        audio_44k = np.frombuffer(data, dtype=np.float32)

        audio_16k = librosa.resample(audio_44k, orig_sr=PI_SAMPLE_RATE, target_sr=MODEL_SAMPLE_RATE)
        model_input = audio_16k.reshape(1, 1, -1).astype(np.float32)
        enhanced_16k = sess.run([output_name], {input_name: model_input})[0].reshape(-1)
        enhanced_44k = librosa.resample(enhanced_16k, orig_sr=MODEL_SAMPLE_RATE, target_sr=PI_SAMPLE_RATE)

        ai_send_sock.sendto(enhanced_44k.astype(np.float32).tobytes(), (PI_IP, AI_SEND_PORT))


def playback_loop():
    """Job 2: receive the FINAL NLMS-cleaned audio from the Pi, play it on Mac speakers."""
    print(f"[Playback] Listening on port {FINAL_LISTEN_PORT} for final audio to play")

    stream = sd.OutputStream(samplerate=PI_SAMPLE_RATE, channels=1, dtype="float32")
    stream.start()

    while True:
        data, _ = final_recv_sock.recvfrom(BYTES_PER_CHUNK + 1024)
        audio = np.frombuffer(data, dtype=np.float32)
        stream.write(audio)


threading.Thread(target=ai_inference_loop, daemon=True).start()
threading.Thread(target=playback_loop, daemon=True).start()

print("Mac server running both jobs. Press Ctrl+C to stop.")
try:
    while True:
        import time
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopped.")