import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report


# Set the working directory to the project's root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(project_root)

# Paths
FEATURES_DIR = "data/features"
MODEL_DIR = "data/model"

N_MFCC = 13

def load_features():
    """Load extracted features and labels."""
    features = np.load(os.path.join(FEATURES_DIR, "features.npy"))
    labels = np.load(os.path.join(FEATURES_DIR, "labels.npy"))
    return features, labels

def train_neural_network(epochs=100, batch_size=32):
    """Train a neural network on the features."""
    # Load features and labels
    features, labels = load_features()

    # One-hot encode labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)

    # Build the neural network
    model = Sequential([
        Dense(256, input_shape=(N_MFCC,), activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(labels.shape[1], activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    print("Training the neural network...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Save the model
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    model.save(os.path.join(MODEL_DIR, "genre_classifier_nn.h5"))
    print(f"Neural network model saved to: {os.path.join(MODEL_DIR, 'genre_classifier_nn.h5')}")

    # Evaluate the model
    print("Evaluating the model...")
    y_pred = model.predict(X_test)

    # Convert predictions to class labels
    y_pred_indices = np.argmax(y_pred, axis=1)
    y_pred_labels = lb.classes_[y_pred_indices]

    # Convert true labels from one-hot encoded to class labels
    y_test_indices = np.argmax(y_test, axis=1)
    y_test_labels = lb.classes_[y_test_indices]

    print(classification_report(y_test_labels, y_pred_labels))

if __name__ == "__main__":
    train_neural_network()
