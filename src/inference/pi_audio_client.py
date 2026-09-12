"""
Runs on the Raspberry Pi. Captures mic input only (no local playback needed
anymore -- final audio plays on the Mac instead).

Flow: mic -> send to Mac (AI) -> receive AI-enhanced audio back ->
      run C-accelerated NLMS -> send FINAL result back to Mac to play.

Requires: pip3 install sounddevice numpy --break-system-packages
Also requires nlms.so already compiled.
"""

import ctypes
import os
import socket
import threading

import numpy as np
import sounddevice as sd

MAC_IP = "169.254.220.173"

AI_SEND_PORT = 5005          # Pi sends mic audio to Mac here
AI_LISTEN_PORT = 5006        # Pi listens here for AI-enhanced audio FROM Mac
FINAL_SEND_PORT = 5007       # Pi sends the final NLMS-cleaned audio to Mac here

SAMPLE_RATE = 44100
CHUNK_SAMPLES = 4410
INPUT_DEVICE = 1

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nlms.so")
nlms = ctypes.CDLL(lib_path)
nlms.nlms_process_block.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
nlms.nlms_process_block.restype = None

ai_send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ai_recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ai_recv_sock.bind(("0.0.0.0", AI_LISTEN_PORT))
final_send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

latest_enhanced = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
buffer_lock = threading.Lock()


def receiver_thread():
    """Background thread: receive AI-enhanced audio from the Mac,
    run NLMS on it, send the final result back to the Mac to play."""
    global latest_enhanced
    while True:
        data, _ = ai_recv_sock.recvfrom(CHUNK_SAMPLES * 4 + 1024)
        arr = np.frombuffer(data, dtype=np.float32).copy()
        if len(arr) != CHUNK_SAMPLES:
            if len(arr) < CHUNK_SAMPLES:
                arr = np.pad(arr, (0, CHUNK_SAMPLES - len(arr)))
            else:
                arr = arr[:CHUNK_SAMPLES]

        # run NLMS on the AI-enhanced audio
        block = np.ascontiguousarray(arr, dtype=np.float32)
        ptr = block.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        nlms.nlms_process_block(ptr, len(block))

        # send the FINAL cleaned result back to the Mac for playback
        final_send_sock.sendto(block.tobytes(), (MAC_IP, FINAL_SEND_PORT))


def mic_callback(indata, frames, time_info, status):
    if status:
        print(status)
    mic_block = np.ascontiguousarray(indata[:, 0], dtype=np.float32)
    ai_send_sock.sendto(mic_block.tobytes(), (MAC_IP, AI_SEND_PORT))


threading.Thread(target=receiver_thread, daemon=True).start()

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    blocksize=CHUNK_SAMPLES,
    channels=1,
    dtype="float32",
    device=INPUT_DEVICE,
    callback=mic_callback,
):
    print("Pi client running: mic -> Mac (AI) -> Pi (NLMS) -> back to Mac speaker.")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        print("Stopped.")