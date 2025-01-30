# Music Genre Classifier

A machine learning project that classifies music tracks into different genres using the GTZAN dataset.

## Installation

### Prerequisites
- Python 3.x
- Git (optional)

### Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/ErnestBoz/Music-genre-classifier.git
   cd music-genre-classifier
   ```
   
2. **Create Data folder**
   ```bash
   mkdir data
   ```

3. **Download GTZAN dataset**
   Download the GTZAN dataset.
   After downloading, place the genres_original folder into the data folder.
   The final folder structure should look like this:

   ```markdown
   Music-genre-classifier/
      ├── data/
      │   └── genres_original/
      │       ├── blues/
      │       ├── classical/
      │       └── ... (other genres)
      └── ...
    ```

4. **Create and activate virual environment**
   In the root directory of the project, create a virtual environment:
   ```bash
   python -m venv .venv
   ```
   
   Activate the virtual environment
   ```bash
   .venv\Scripts\Activate
   ```

5. **Install dependencies**
   After activating the virtual environment, install the dependencies from the   requirements.txt file:
   ```bash
   pip install -r requirements.txt
   ```

6. **Run src files:**
   ```python
   python src/preprocess.py
   python src/feature_extraction.py
   python src/train_model.py
   ```

## Predicting the genre

  In the predict_genre.py script, line 52, update the path to your .wav file.
  ```python
  test_audio = "data/genres_original/blues/blues.00090.wav"
  ```
  After updating the path, run the prediction:
  ```bash
  python src/predict_genre.py
  ```

## File description:
- preprocess.py: Processes the GTZAN audio files for further feature extraction.
- feature_extraction.py: Extracts audio features (e.g., MFCC) from the processed audio files.
- train_model.py: Trains the model using the extracted features.
- predict_genre.py: Predicts the genre of a given audio file.


