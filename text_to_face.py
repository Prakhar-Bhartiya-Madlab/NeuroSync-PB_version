import requests
import pygame
import torch
import os
import uuid
from threading import Thread, Event, Lock

from generate_face_shapes import generate_facial_data_from_bytes
from livelink.connect.livelink_init import create_socket_connection, initialize_py_face
from livelink.animations.default_animation import default_animation_loop, stop_default_animation
from livelink.send_to_unreal import pre_encode_facial_data, send_pre_encoded_data_to_unreal
from utils.audio.play_audio import play_audio_from_memory
from utils.audio.save_audio import save_audio_file
from utils.csv.save_csv import save_generated_data_as_csv
from model import load_model

# Configuration
config = {
    'sr': 88200,
    'frame_rate': 60,
    'hidden_dim': 1024,
    'n_layers': 4,
    'num_heads': 4,
    'dropout': 0.0,
    'output_dim': 68,
    'input_dim': 26 + 26 + 26,
    'frame_size': 256,
}

model_path = '_out/model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model at the start
print("Loading model...")
model = load_model(model_path, config, device)
print("Model loaded successfully.")

queue_lock = Lock()

# ElevenLabs API configuration
ELEVENLABS_API_KEY = ""  # Replace with your actual ElevenLabs API key
VOICE_ID = ""
API_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream"

GENERATED_DIR = 'generated'

def initialize_directories():
    if not os.path.exists(GENERATED_DIR):
        os.makedirs(GENERATED_DIR)

def get_elevenlabs_audio(text):
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "text": text,
        "model_id": "eleven_turbo_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.5,
            "use_speaker_boost": True
        }
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()

    audio_data = response.content
    return audio_data

def preprocess_audio(audio_bytes, model, device):
    # Generate facial data from the audio bytes directly
    return generate_facial_data_from_bytes(audio_bytes, model, device, config)

def save_generated_data(audio_bytes, generated_facial_data):
    unique_id = str(uuid.uuid4())
    output_dir = os.path.join(GENERATED_DIR, unique_id)
    os.makedirs(output_dir, exist_ok=True)

    audio_path = os.path.join(output_dir, 'audio.wav')
    shapes_path = os.path.join(output_dir, 'shapes.csv')

    save_audio_file(audio_bytes, audio_path)
    save_generated_data_as_csv(generated_facial_data, shapes_path)

    return unique_id, audio_path, shapes_path

def run_audio_animation(audio_bytes, generated_facial_data, py_face, socket_connection, default_animation_thread):
    with queue_lock:
        stop_default_animation.set()
        if default_animation_thread and default_animation_thread.is_alive():
            default_animation_thread.join()

    start_event = Event()

    # Pre-encode the facial data
    encoded_facial_data = pre_encode_facial_data(generated_facial_data, py_face)

    # Create the threads for audio and animation playback
    audio_thread = Thread(target=play_audio_from_memory, args=(audio_bytes, start_event))
    data_thread = Thread(target=send_pre_encoded_data_to_unreal, args=(encoded_facial_data, start_event, 60, socket_connection))

    # Start the threads
    audio_thread.start()
    data_thread.start()
    
    # Trigger the start event
    start_event.set()
    
    # Wait for both threads to finish
    audio_thread.join()
    data_thread.join()

    # Restart the default animation
    with queue_lock:
        stop_default_animation.clear()
        default_animation_thread = Thread(target=default_animation_loop, args=(py_face,))
        default_animation_thread.start()

if __name__ == "__main__":
    # Initialize directories
    initialize_directories()

    # Initialize PyFace and the default animation
    py_face = initialize_py_face()
    socket_connection = create_socket_connection()

    default_animation_thread = Thread(target=default_animation_loop, args=(py_face,))
    default_animation_thread.start()

    try:
        while True:
            text_input = input("Enter the text to generate speech (or 'q' to quit): ").strip()
            if text_input.lower() == 'q':
                break
            elif text_input:
                # Step 1: Generate audio using ElevenLabs API
                audio_bytes = get_elevenlabs_audio(text_input)

                # Step 2: Generate facial data from the audio bytes
                generated_facial_data = preprocess_audio(audio_bytes, model, device)

                # Step 4: Play both the generated facial shapes and the audio
                run_audio_animation(audio_bytes, generated_facial_data, py_face, socket_connection, default_animation_thread)
                
                # Step 3: Save the generated data (swapped as this is just so you have a record of what was generated, no need to hold up the playback...)
                save_generated_data(audio_bytes, generated_facial_data)
            
            else:
                print("No text provided.")
    finally:
        # Clean up and close resources
        stop_default_animation.set()
        if default_animation_thread:
            default_animation_thread.join()
        pygame.quit()
        socket_connection.close()
