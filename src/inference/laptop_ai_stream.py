import numpy as np
import onnxruntime as ort
import sounddevice as sd
from pathlib import Path

# --- Configuration ---
# Dynamically find the project root so it runs from any terminal directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Fallback to the unquantized model for maximum accuracy during the test
MODEL_PATH = str(PROJECT_ROOT / "models_exported" / "sih_anc_model.onnx")

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 4000  # 250ms chunks for low latency

print(f"Loading model from: {MODEL_PATH}...")
try:
    sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print("✅ Model loaded successfully. Starting stream...")
except Exception as e:
    print(f"❌ Failed to load model. Error: {e}")
    exit(1)
    
def audio_callback(indata, outdata, frames, time_info, status):
    if status:
        print(f"Audio Status: {status}")
        
    audio_in = np.squeeze(indata).astype(np.float32)
    
    if len(audio_in) < CHUNK_SAMPLES:
        audio_in = np.pad(audio_in, (0, CHUNK_SAMPLES - len(audio_in)))
        
    # --- Dynamic Volume Normalization ---
    # Scale the mic input up to 1.0 so the model recognizes the noise
    max_amp = np.max(np.abs(audio_in)) + 1e-8
    normalized_audio = audio_in / max_amp
    
    ort_inputs = {input_name: normalized_audio.reshape(1, 1, -1)}
    
    try:
        ort_outs = sess.run(None, ort_inputs)
        clean_audio = ort_outs[0].flatten()
        
        # --- Restore Original Envelope ---
        # Scale the cleaned audio back down to match your mic's original volume
        clean_audio = clean_audio * max_amp
        
    except Exception as e:
        # Explicitly print the error so it doesn't fail silently
        print(f"\n🛑 INFERENCE ERROR: {e}\n")
        clean_audio = audio_in
        
    outdata[:] = clean_audio[:frames].reshape(-1, 1)

# --- Start Stream ---
print("\n🎧 CRITICAL: PLUG IN WIRED HEADPHONES NOW!")
print("Using laptop speakers will cause a deafening feedback loop.")
input("Press Enter when your headphones are connected...")

print("\n🚀 Streaming live. Press Ctrl+C to stop.")
try:
    with sd.Stream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=CHUNK_SAMPLES):
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("\n🛑 Stream stopped.")
    
    
        