from flask import Flask, request, jsonify
import tensorflow as tf
import librosa
import numpy as np
import os
import logging

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "final_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Universal preprocessing function
def preprocess_audio(file_path, target_sample_rate=22050, n_mfcc=13, fixed_time_steps=87):
    """
    Preprocess an audio file to make it compatible with the model.
    
    Args:
        file_path (str): Path to the audio file.
        target_sample_rate (int): Target sample rate for resampling.
        n_mfcc (int): Number of MFCC coefficients to extract.
        fixed_time_steps (int): Fixed number of time steps for MFCCs.
    
    Returns:
        np.array: Preprocessed audio data with shape (1, fixed_time_steps, n_mfcc).
    """
    # Load the audio file and resample to target sample rate
    audio, sr = librosa.load(file_path, sr=target_sample_rate)
    
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs = mfccs.T  # Transpose to (time_steps, n_mfcc)
    
    # Pad or truncate to fixed number of time steps
    if mfccs.shape[0] < fixed_time_steps:
        # Pad with zeros if shorter than fixed_time_steps
        pad_width = ((0, fixed_time_steps - mfccs.shape[0]), (0, 0))
        mfccs = np.pad(mfccs, pad_width, mode='constant')
    else:
        # Truncate if longer than fixed_time_steps
        mfccs = mfccs[:fixed_time_steps, :]
    
    # Expand dimensions to match model input shape
    mfccs = np.expand_dims(mfccs, axis=0)  # Add batch dimension
    
    return mfccs

@app.route('/')
def home():
    return "Gunshot Detection Backend is running!"

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an audio file is included
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    file_path = "temp_audio.wav"
    file.save(file_path)

    try:
        # Preprocess the audio
        input_data = preprocess_audio(file_path)
        print(f"Preprocessed data shape: {input_data.shape}")  # Debugging
        
        # Make prediction
        prediction = model.predict(input_data)
        result = int(prediction[0] > 0.5)  # Output 1 for gunshot, 0 for no gunshot
        return jsonify({"result": result})
    except Exception as e:
        print(f"Error during prediction: {str(e)}")  # Debugging
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use PORT if set, otherwise default to 5000
    app.run(host='0.0.0.0', port=port)
