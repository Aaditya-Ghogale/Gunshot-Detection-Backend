import os
import random
import subprocess

# Path to the folder containing test audio files
TEST_AUDIO_FOLDER = r"C:\Users\aadit\Downloads\papap\test"

# Flask server URL
FLASK_SERVER_URL = "http://127.0.0.1:5000/predict"

def get_random_audio_file(folder):
    """
    Get a random audio file from the specified folder.
    
    Args:
        folder (str): Path to the folder containing audio files.
    
    Returns:
        str: Full path to the randomly selected audio file.
    """
    # List all files in the folder
    audio_files = [f for f in os.listdir(folder) if f.endswith('.wav')]
    
    if not audio_files:
        raise FileNotFoundError("No .wav files found in the folder.")
    
    # Select a random file
    random_file = random.choice(audio_files)
    return os.path.join(folder, random_file)

def send_audio_to_server(file_path):
    """
    Send an audio file to the Flask server using curl.
    
    Args:
        file_path (str): Path to the audio file.
    """
    # Construct the curl command with the --silent flag to suppress progress output
    curl_command = [
        'curl', '--silent', '-X', 'POST',
        '-F', f'file=@{file_path}',
        FLASK_SERVER_URL
    ]
    
    # Run the curl command
    result = subprocess.run(curl_command, capture_output=True, text=True)
    
    # Print the server's response
    print("Server Response:")
    print(result.stdout)
    
    # Print errors (if any)
    if result.stderr:
        print("Curl Error:")
        print(result.stderr)

if __name__ == '__main__':
    try:
        # Get a random audio file
        random_audio = get_random_audio_file(TEST_AUDIO_FOLDER)
        print(f"Selected audio file: {random_audio}")
        
        # Send the audio file to the Flask server
        send_audio_to_server(random_audio)
    except Exception as e:
        print(f"Error: {e}")