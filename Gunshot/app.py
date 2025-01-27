import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import librosa
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
MODEL_PATH = "final_model.keras"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

def preprocess_audio(file_path, target_sample_rate=22050, n_mfcc=13, fixed_time_steps=87):
    try:
        audio, sr = librosa.load(file_path, sr=target_sample_rate)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfccs = mfccs.T
        
        if mfccs.shape[0] < fixed_time_steps:
            pad_width = ((0, fixed_time_steps - mfccs.shape[0]), (0, 0))
            mfccs = np.pad(mfccs, pad_width, mode='constant')
        else:
            mfccs = mfccs[:fixed_time_steps, :]
            
        mfccs = np.expand_dims(mfccs, axis=0)
        return mfccs
    except Exception as e:
        raise Exception(f"Error in preprocessing: {str(e)}")

@app.route('/')
def home():
    return "Gunshot Detection Backend is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
        
    if not file.filename.endswith('.wav'):
        return jsonify({"error": "Only .wav files are supported"}), 400

    file_path = "temp_audio.wav"
    try:
        file.save(file_path)
        input_data = preprocess_audio(file_path)
        print(f"Preprocessed data shape: {input_data.shape}")
        
        prediction = model.predict(input_data)
        result = int(prediction[0] > 0.5)
        
        return jsonify({
            "result": result,
            "confidence": float(prediction[0]),
            "message": "Gunshot detected" if result == 1 else "No gunshot detected"
        })
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500
        
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
