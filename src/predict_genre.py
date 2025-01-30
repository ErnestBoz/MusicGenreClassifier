import os
import librosa
import joblib
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Paths
FEATURES_DIR = "data/features"
MODEL_DIR = "data/model"

N_MFCC = 13

# Set the working directory to the project's root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(project_root)

def predict_genre(audio_path):
    
    # Load the trained model
    model_path = os.path.join(MODEL_DIR, "genre_classifier_nn.h5")
    model = load_model(model_path)


    # Load genre labels
    labels_path = os.path.join(FEATURES_DIR, "../processed/genre_labels.pkl")
    with open(labels_path, "rb") as f:
        genre_labels = pickle.load(f)
    label_to_genre = {v: k for k, v in genre_labels.items()}  # Reverse mapping
    
    # Load and preprocess the audio file
    print(f"Processing audio file: {audio_path}")
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    y = librosa.util.fix_length(y, size=int(22050 * 3))  # Truncate/pad to 3 seconds
    
    # Extract features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    
    # Predict genre
    mfcc_mean = mfcc_mean.reshape(1, -1)  # Reshape for sklearn
    predicted_label = np.argmax(model.predict(mfcc_mean), axis=-1)[0]


    predicted_genre = label_to_genre[predicted_label]
    
    print(f"Predicted Genre: {predicted_genre}")
    return predicted_genre

if __name__ == "__main__":
    # Test the model on a sample audio file
    test_audio = "some path"  # Replace with your file path
    predict_genre(test_audio)
