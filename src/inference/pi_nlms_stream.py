"""
Pi 3B side: audio-in (from laptop via cable) -> NLMS residual filter -> speaker-out.

Run this ON the Raspberry Pi 3B.

Requires: pip install sounddevice numpy
List audio devices first with: python3 -c "import sounddevice as sd; print(sd.query_devices())"
to find your USB audio adapter's device index, then set INPUT_DEVICE / OUTPUT_DEVICE below.
"""

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
BLOCK_SIZE = 1024       # smaller block = lower latency, but more CPU overhead per block
FILTER_LEN = 32         # NLMS filter taps — keep small, Pi 3B is weak
MU = 0.5                # NLMS step size (0 < mu <= 1), higher = faster adapt, less stable

INPUT_DEVICE = None      # set to your USB audio adapter's index once you check sd.query_devices()
OUTPUT_DEVICE = None

# NLMS filter state (persists across callback calls)
w = np.zeros(FILTER_LEN, dtype=np.float32)
x_hist = np.zeros(FILTER_LEN, dtype=np.float32)


def nlms_step(x_n: float, d_n: float, eps: float = 1e-6):
    """One NLMS update step.

    x_n: current input sample (reference/noisy sample)
    d_n: desired sample (here, same signal — NLMS adapts to predict and
         subtract residual correlated noise structure)
    Returns the filtered (error) output sample.
    """
    global w, x_hist
    x_hist = np.roll(x_hist, 1)
    x_hist[0] = x_n

    y_n = np.dot(w, x_hist)
    e_n = d_n - y_n

    norm = np.dot(x_hist, x_hist) + eps
    w += (MU / norm) * e_n * x_hist

    return e_n


def audio_callback(indata, outdata, frames, time_info, status):
    if status:
        print(status)

    block = indata[:, 0].astype(np.float32)
    out_block = np.zeros_like(block)

    for i in range(len(block)):
        out_block[i] = nlms_step(block[i], block[i])

    outdata[:, 0] = out_block


with sd.Stream(
    samplerate=SAMPLE_RATE,
    blocksize=BLOCK_SIZE,
    channels=1,
    dtype="float32",
    device=(INPUT_DEVICE, OUTPUT_DEVICE),
    callback=audio_callback,
):
    print("Pi NLMS stream running. Press Ctrl+C to stop.")
    try:
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        print("Stopped.")