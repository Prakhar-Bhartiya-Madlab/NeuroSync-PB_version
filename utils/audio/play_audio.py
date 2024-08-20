# This code is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# For more details, visit: https://creativecommons.org/licenses/by-nc/4.0/

import pygame 
import time
import io


def play_audio_bytes(audio_bytes, start_event):
    try:
        pygame.mixer.init()
        audio_file = io.BytesIO(audio_bytes)
        pygame.mixer.music.load(audio_file)
        
        start_event.wait()
        pygame.mixer.music.play()

        chunk_size = 100
        while pygame.mixer.music.get_busy():
            time.sleep(chunk_size / 1000.0)
            pygame.time.Clock().tick(10)
    except pygame.error as e:
        if "Unknown WAVE format" in str(e):
            print("Unknown WAVE format encountered. Skipping to the next item in the queue.")
        else:
            print(f"Error in play_audio: {e}")
    except Exception as e:
        print(f"Error in play_audio: {e}")
        
def play_audio_from_memory(audio_data, start_event):
    try:
        pygame.mixer.init()
        audio_file = io.BytesIO(audio_data)
        pygame.mixer.music.load(audio_file)
        
        start_event.wait()
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except pygame.error as e:
        if "Unknown WAVE format" in str(e):
            print("Unknown WAVE format encountered. Skipping to the next item in the queue.")
        else:
            print(f"Error in play_audio_from_memory: {e}")
    except Exception as e:
        print(f"Error in play_audio_from_memory: {e}")
        
def play_audio_from_path(audio_path, start_event):
    pygame.mixer.init()
    pygame.mixer.music.load(audio_path)
    start_event.wait()
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)