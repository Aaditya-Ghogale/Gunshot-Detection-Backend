import os
from flask import Flask, request, jsonify
import tensorflow as tf
import librosa
import numpy as np

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "final_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Universal preprocessing function
def preprocess_audio(file_path, target_sample_rate=22050, n_mfcc=13, fixed_time_steps=87):
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

@app.route('/')
def home():
    return "Gunshot Detection Backend is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    file_path = "temp_audio.wav"
    file.save(file_path)

    try:
        input_data = preprocess_audio(file_path)
        print(f"Preprocessed data shape: {input_data.shape}")
        prediction = model.predict(input_data)
        result = int(prediction[0] > 0.5)
        return jsonify({"result": result})
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Use PORT if set, otherwise default to 10000
    app.run(host='0.0.0.0', port=port)
