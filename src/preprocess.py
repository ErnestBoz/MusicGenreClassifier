import os
import librosa
import numpy as np
import pickle

DATA_DIR = "data/genres_original"  # Path to your dataset
PROCESSED_DIR = "data/processed"   # Path to save processed data
SAMPLE_RATE = 22050                # Standard sampling rate
MAX_DURATION = 3                   # Standardize to 3 seconds
N_MFCC = 13                        # Number of MFCC features


def load_audio(file_path):
    """Load an audio file, convert to mono, and resample."""
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    return y, sr


def standardize_audio(y):
    """Truncate or pad the audio to a fixed length."""
    fixed_length = int(SAMPLE_RATE * MAX_DURATION)
    if len(y) > fixed_length:
        return y[:fixed_length]  # Truncate
    else:
        return np.pad(y, (0, fixed_length - len(y)))  # Pad


def preprocess_genres():
    """Iterate through dataset and preprocess all files."""
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
    
    genre_labels = {}
    audio_data = []
    labels = []
    
    for idx, genre in enumerate(os.listdir(DATA_DIR)):
        genre_folder = os.path.join(DATA_DIR, genre)
        if os.path.isdir(genre_folder):
            print(f"Processing genre: {genre}")
            for file_name in os.listdir(genre_folder):
                file_path = os.path.join(genre_folder, file_name)
                try:
                    y, sr = load_audio(file_path)
                    y = standardize_audio(y)
                    audio_data.append(y)
                    labels.append(idx)  # Label corresponds to genre index
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
            
            # Map genre name to label
            genre_labels[genre] = idx
    
    # Save preprocessed audio and labels
    np.save(os.path.join(PROCESSED_DIR, "audio_data.npy"), audio_data)
    np.save(os.path.join(PROCESSED_DIR, "labels.npy"), labels)
    
    # Save genre-label mapping
    with open(os.path.join(PROCESSED_DIR, "genre_labels.pkl"), "wb") as f:
        pickle.dump(genre_labels, f)

    print("Preprocessing complete. Data saved in:", PROCESSED_DIR)


if __name__ == "__main__":
    preprocess_genres()
