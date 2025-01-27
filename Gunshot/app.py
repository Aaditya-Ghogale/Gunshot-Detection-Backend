import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import librosa
import numpy as np
import uuid
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model at startup
try:
    MODEL_PATH = "final_model.keras"
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

# Create upload folder
UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_audio(file_path, target_sample_rate=22050, n_mfcc=13, fixed_time_steps=87):
    """Optimized audio preprocessing"""
    try:
        # Load audio with a shorter duration if possible
        audio, sr = librosa.load(file_path, sr=target_sample_rate, duration=10)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfccs = mfccs.T

        if mfccs.shape[0] < fixed_time_steps:
            pad_width = ((0, fixed_time_steps - mfccs.shape[0]), (0, 0))
            mfccs = np.pad(mfccs, pad_width, mode='constant')
        else:
            mfccs = mfccs[:fixed_time_steps, :]

        return np.expand_dims(mfccs, axis=0)
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise

@app.route('/')
def home():
    return "Gunshot Detection Backend is running!"

@app.route('/predict', methods=['POST'])
def predict():
    file_path = None
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Save file with unique name
        unique_filename = f"{uuid.uuid4()}.wav"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(file_path)
        logger.info(f"File saved: {file_path}")

        # Process audio
        input_data = preprocess_audio(file_path)
        logger.info(f"Audio preprocessed, shape: {input_data.shape}")

        # Make prediction
        prediction = model.predict(input_data, verbose=0)
        result = float(prediction[0][0])

        response = {
            "result": int(result > 0.5),
            "confidence": float(result),
            "message": "Gunshot detected" if result > 0.5 else "No gunshot detected"
        }
        logger.info(f"Prediction complete: {response}")

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

    finally:
        # Cleanup
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Temporary file removed: {file_path}")
            except Exception as e:
                logger.error(f"Error removing temporary file: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
