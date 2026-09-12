import socket
import numpy as np
import sounddevice as sd
import queue
import threading

# --- Config ---
PI_HOSTNAME = "raspberrypi.local"
PORT = 5005
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 4000
PAYLOAD_SIZE = CHUNK_SAMPLES * 4

mic_queue = queue.Queue(maxsize=2)
speaker_queue = queue.Queue(maxsize=2)

# --- TCP Setup ---
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print(f"Connecting to Pi AI TCP Server at {PI_HOSTNAME}:{PORT}...")
sock.connect((PI_HOSTNAME, PORT))
print("✅ Connected!")

def mic_callback(indata, frames, time_info, status):
    try:
        mic_queue.put_nowait(indata.copy())
    except queue.Full:
        pass

def speaker_callback(outdata, frames, time_info, status):
    try:
        clean_audio = speaker_queue.get_nowait()
        outdata[:] = clean_audio[:frames].reshape(-1, 1)
    except queue.Empty:
        outdata.fill(0)

def recv_exact(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet: return None
        data.extend(packet)
    return bytes(data)

def network_loop():
    while True:
        try:
            # Grab mic data and send
            mic_data = mic_queue.get()
            payload = np.squeeze(mic_data).astype(np.float32).tobytes()
            sock.sendall(payload)

            # Wait for exact 16KB payload from Pi
            data = recv_exact(sock, PAYLOAD_SIZE)
            if data:
                clean_audio = np.frombuffer(data, dtype=np.float32)
                try:
                    speaker_queue.put_nowait(clean_audio)
                except queue.Full:
                    pass
        except Exception as e:
            print(f"\n❌ Network error: {e}")
            break

net_thread = threading.Thread(target=network_loop, daemon=True)
net_thread.start()

# --- Start Stream ---
try:
    in_stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=CHUNK_SAMPLES, callback=mic_callback)
    out_stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=CHUNK_SAMPLES, callback=speaker_callback)
    
    with in_stream, out_stream:
        print("\n🚀 Streaming live! Speak into your Mac mic.")
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("\n🛑 Stream stopped.")
finally:
    sock.close()