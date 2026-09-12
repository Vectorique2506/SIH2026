import socket
import sounddevice as sd
import numpy as np

PI_IP = "raspberrypi.local"
UDP_PORT = 5005
SAMPLE_RATE = 16000
BLOCK_SIZE = 512

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# The crucial fix: Timeout MUST be lower than block duration (32ms)
sock.settimeout(0.01) # 10ms maximum wait time

def callback(indata, outdata, frames, time, status):
    # 1. Send Mac Mic audio to Pi
    sock.sendto(indata[:, 0].tobytes(), (PI_IP, UDP_PORT))
    
    # 2. Receive Filtered audio back from Pi
    try:
        data, _ = sock.recvfrom(4096)
        outdata[:, 0] = np.frombuffer(data, dtype=np.float32)
    except (socket.timeout, BlockingIOError):
        # If the Wi-Fi lags, play silence instantly instead of overflowing
        outdata.fill(0)
        print("x", end="", flush=True) 

print(f"Mac streaming to Pi Edge Node at {PI_IP}. Press Ctrl+C to stop.")
with sd.Stream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=1, dtype='float32', callback=callback):
    sd.sleep(100000)
