import os
import json
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd

AUDIO_FILE = "Free.mp3"
SPEC_OUT = "free_spectrogram.png"
FEATURES_OUT = "free_features.csv"
FEATURES_JSON = "free_features.json"

def load_audio(path, sr=22050, fallback_seconds=5):
    if os.path.isfile(path):
        y, sr = librosa.load(path, sr=sr, mono=True)
        return y, sr
    # fallback: sine wave (useful for testing if file missing)
    T = fallback_seconds
    t = np.linspace(0, T, int(T * sr), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    return y, sr

def save_amplitude_spectrogram(y, sr, out_path=SPEC_OUT, n_fft=2048, hop_length=512):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Amplitude Spectrogram (dB)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path

def extract_features(y, sr, hop_length=512, n_mfcc=13):
    feats = {}

    # Time-domain
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
    feats['zcr_mean'] = float(np.mean(zcr))
    feats['zcr_std'] = float(np.std(zcr))

    rmse = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    feats['rmse_mean'] = float(np.mean(rmse))
    feats['rmse_std'] = float(np.std(rmse))

    # Spectral
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    feats['spectral_centroid_mean'] = float(np.mean(centroid))
    feats['spectral_centroid_std'] = float(np.std(centroid))

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
    feats['spectral_bandwidth_mean'] = float(np.mean(bandwidth))
    feats['spectral_bandwidth_std'] = float(np.std(bandwidth))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
    feats['spectral_rolloff_mean'] = float(np.mean(rolloff))
    feats['spectral_rolloff_std'] = float(np.std(rolloff))

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    for i in range(n_mfcc):
        feats[f'mfcc{i+1}_mean'] = float(np.mean(mfcc[i]))
        feats[f'mfcc{i+1}_std'] = float(np.std(mfcc[i]))

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    for i in range(chroma.shape[0]):
        feats[f'chroma{i+1}_mean'] = float(np.mean(chroma[i]))
        feats[f'chroma{i+1}_std'] = float(np.std(chroma[i]))

    return feats

def save_features_csv(feats: dict, out_path=FEATURES_OUT):
    df = pd.DataFrame([feats])
    df.to_csv(out_path, index=False)
    return out_path

def save_features_json(feats: dict, out_path=FEATURES_JSON):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(feats, f, ensure_ascii=False, indent=2)
    return out_path

def main():
    y, sr = load_audio(AUDIO_FILE)
    spec_path = save_amplitude_spectrogram(y, sr)
    feats = extract_features(y, sr)
    csv_path = save_features_csv(feats)
    json_path = save_features_json(feats)

    print(f"Spectrogram saved: {spec_path}")
    print(f"Features saved (CSV): {csv_path}")
    print(f"Features saved (JSON): {json_path}")

if __name__ == "__main__":
    main()