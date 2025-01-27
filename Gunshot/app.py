import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import librosa
import numpy as np
import uuid

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
MODEL_PATH = "final_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Create a temporary directory if it doesn't exist
UPLOAD_FOLDER = 'temp_uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def preprocess_audio(file_path, target_sample_rate=22050, n_mfcc=13, fixed_time_steps=87):
    """Preprocess audio file for prediction"""
    try:
        # Load and preprocess the audio file
        audio, sr = librosa.load(file_path, sr=target_sample_rate)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfccs = mfccs.T

        # Pad or truncate to fixed length
        if mfccs.shape[0] < fixed_time_steps:
            pad_width = ((0, fixed_time_steps - mfccs.shape[0]), (0, 0))
            mfccs = np.pad(mfccs, pad_width, mode='constant')
        else:
            mfccs = mfccs[:fixed_time_steps, :]

        # Add batch dimension
        mfccs = np.expand_dims(mfccs, axis=0)
        return mfccs
    except Exception as e:
        raise Exception(f"Error in preprocessing: {str(e)}")

@app.route('/')
def home():
    return "Gunshot Detection Backend is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Generate a unique filename
        unique_filename = f"{uuid.uuid4()}.wav"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)

        try:
            # Save the file
            file.save(file_path)
            print(f"File saved temporarily as: {file_path}")

            # Preprocess the audio
            input_data = preprocess_audio(file_path)
            print(f"Preprocessed data shape: {input_data.shape}")

            # Make prediction
            prediction = model.predict(input_data)
            result = float(prediction[0][0])  # Convert to Python float
            
            # Prepare response
            response = {
                "result": int(result > 0.5),
                "confidence": float(result),
                "message": "Gunshot detected" if result > 0.5 else "No gunshot detected"
            }

            return jsonify(response)

        except Exception as e:
            print(f"Error during processing: {str(e)}")
            return jsonify({"error": str(e)}), 500

        finally:
            # Clean up: remove the temporary file
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Temporary file removed: {file_path}")

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
