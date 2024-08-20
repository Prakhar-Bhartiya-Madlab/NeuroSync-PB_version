# regen_generated.py

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shutil
import uuid

from utils.generate_face_shapes import generate_facial_data_from_bytes 
from utils.csv.save_csv import save_generated_data_as_csv

from api.api_config import model, device

GENERATED_DIR = 'generated/generated'

def process_audio_files():
    directories = [d for d in os.listdir(GENERATED_DIR) if os.path.isdir(os.path.join(GENERATED_DIR, d))]
    
    for directory in directories:
        dir_path = os.path.join(GENERATED_DIR, directory)
        audio_path = os.path.join(dir_path, 'audio.wav')
        shapes_path = os.path.join(dir_path, 'shapes.csv')
        
        if os.path.exists(audio_path):
            print(f"Processing: {audio_path}")
            
            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()
            
            generated_facial_data = generate_facial_data_from_bytes(audio_bytes, model, device)
            
            old_dir = os.path.join(dir_path, 'old')
            os.makedirs(old_dir, exist_ok=True)

            if os.path.exists(shapes_path):
                unique_old_name = f"shapes_{uuid.uuid4()}.csv"
                shutil.move(shapes_path, os.path.join(old_dir, unique_old_name))
            
            save_generated_data_as_csv(generated_facial_data, shapes_path)
            
            print(f"New shapes.csv generated and old shapes.csv moved to {old_dir}")

if __name__ == '__main__':
    process_audio_files()
