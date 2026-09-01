"""
Mic-in -> quantized ONNX model -> headset-out streaming loop.

Owner: hardware track. Runs ON the Raspberry Pi 5.

TODO:
- Use sounddevice or pyaudio to open an input stream from the primary mic
  and an output stream to the headset
- Buffer audio into fixed-size chunks matching your model's expected input
- Run onnxruntime.InferenceSession on each chunk
- Write the enhanced chunk to the output stream
- Log per-chunk latency for latency_bench.py to summarize
"""

# import sounddevice as sd
# import onnxruntime as ort
# import numpy as np


def run_stream(onnx_model_path: str, sample_rate: int = 16000, block_size: int = 1024):
    raise NotImplementedError


if __name__ == "__main__":
    pass
