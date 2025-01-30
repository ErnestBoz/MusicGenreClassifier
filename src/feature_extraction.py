import numpy as np
import librosa
import os

# Paths
PROCESSED_DIR = "data/processed"  # Path to preprocessed data
FEATURES_DIR = "data/features"    # Path to save features
N_MFCC = 13                       # Number of MFCC features


def extract_mfcc(audio, sr=22050, n_mfcc=13):
    """Extract MFCC features from an audio signal."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    # Take mean across time axis to reduce dimensionality
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean


def extract_features():
    """Load preprocessed audio and extract features."""
    # Load preprocessed audio and labels
    audio_data = np.load(os.path.join(PROCESSED_DIR, "audio_data.npy"), allow_pickle=True)
    labels = np.load(os.path.join(PROCESSED_DIR, "labels.npy"))
    
    # Ensure output directory exists
    if not os.path.exists(FEATURES_DIR):
        os.makedirs(FEATURES_DIR)
    
    print("Extracting features...")
    features = []
    for idx, audio in enumerate(audio_data):
        try:
            mfcc = extract_mfcc(audio)
            features.append(mfcc)
        except Exception as e:
            print(f"Error extracting features for sample {idx}: {e}")
    
    # Save extracted features and labels
    np.save(os.path.join(FEATURES_DIR, "features.npy"), features)
    np.save(os.path.join(FEATURES_DIR, "labels.npy"), labels)
    print(f"Features saved in: {FEATURES_DIR}")


if __name__ == "__main__":
    extract_features()
