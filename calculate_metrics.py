import numpy as np
import soundfile as sf
from pesq import pesq
from pystoi import stoi

def calculate_snr(clean, enhanced):
    # SNR = 10 * log10 ( Power of Signal / Power of Noise )
    noise = clean - enhanced
    signal_power = np.sum(clean ** 2)
    noise_power = np.sum(noise ** 2) + 1e-8
    return 10 * np.log10(signal_power / noise_power)

def run_evaluation(clean_path, enhanced_path):
    print(f"Loading files:\n- Clean: {clean_path}\n- Enhanced: {enhanced_path}\n")
    
    # Load audio files (assuming they are 16kHz)
    clean, sr1 = sf.read(clean_path)
    enhanced, sr2 = sf.read(enhanced_path)

    # Ensure audio lengths match perfectly
    min_len = min(len(clean), len(enhanced))
    clean = clean[:min_len]
    enhanced = enhanced[:min_len]

    # Calculate STOI (Scale 0.0 to 1.0)
    stoi_score = stoi(clean, enhanced, 16000, extended=False)
    
    # Calculate PESQ (Scale -0.5 to 4.5, 'wb' = wideband 16kHz)
    pesq_score = pesq(16000, clean, enhanced, 'wb')
    
    # Calculate SNR (Decibels)
    snr_score = calculate_snr(clean, enhanced)

    print("📊 FINAL DRDO METRICS 📊")
    print("-" * 25)
    print(f"STOI Score : {stoi_score:.4f} (Intelligibility - closer to 1.0 is better)")
    print(f"PESQ Score : {pesq_score:.4f} (Quality - scale -0.5 to 4.5)")
    print(f"SNR  Score : {snr_score:.2f} dB (Signal strength - higher is better)")
    print("-" * 25)

# --- Run the Test ---
# Replace these strings with your actual test files
run_evaluation("clean_voice.wav", "model_output.wav")