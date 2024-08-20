# This code is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# For more details, visit: https://creativecommons.org/licenses/by-nc/4.0/

import os
import uuid
from threading import Thread, Event, Lock

from livelink.connect.livelink_init import create_socket_connection
from livelink.send_to_unreal import pre_encode_facial_data, send_pre_encoded_data_to_unreal
from livelink.animations.default_animation import default_animation_loop, stop_default_animation

from generate_face_shapes import generate_facial_data_from_bytes

from utils.csv.save_csv import save_generated_data_as_csv

from utils.audio.play_audio import play_audio_bytes
from utils.audio.save_audio import save_audio_file

GENERATED_DIR = 'generated'
queue_lock = Lock()

def initialize_directories():
    if not os.path.exists(GENERATED_DIR):
        os.makedirs(GENERATED_DIR)

def run_audio_animation(audio_bytes, encoded_facial_data, py_face, socket_connection):
    start_event = Event()

    audio_thread = Thread(target=play_audio_bytes, args=(audio_bytes, start_event))
    data_thread = Thread(target=send_pre_encoded_data_to_unreal, args=(encoded_facial_data, start_event, 60, socket_connection))

    audio_thread.start()
    data_thread.start()

    start_event.set()
    audio_thread.join()
    data_thread.join()

def preprocess_audio(audio_bytes, model, device):
    return generate_facial_data_from_bytes(audio_bytes, model, device)

def save_generated_data(audio_bytes, generated_facial_data):
    unique_id = str(uuid.uuid4())
    output_dir = os.path.join(GENERATED_DIR, unique_id)
    os.makedirs(output_dir, exist_ok=True)

    audio_path = os.path.join(output_dir, 'audio.wav')
    shapes_path = os.path.join(output_dir, 'shapes.csv')

    save_audio_file(audio_bytes, audio_path)
    save_generated_data_as_csv(generated_facial_data, shapes_path)

    return unique_id, audio_path, shapes_path

def process_preprocessing_queue(request_queue, preprocessed_data_queue, model, device):
    while True:
        audio_bytes = request_queue.get()
        if audio_bytes is None:
            break
        generated_facial_data = preprocess_audio(audio_bytes, model, device)
        save_generated_data(audio_bytes, generated_facial_data)
        preprocessed_data_queue.put((audio_bytes, generated_facial_data))
        request_queue.task_done()

def process_playback_queue(preprocessed_data_queue, py_face, default_animation_thread, request_queue):
    global stop_default_animation
    while True:
        audio_bytes, generated_facial_data = preprocessed_data_queue.get()
        if audio_bytes is None:
            break

        with queue_lock:
            stop_default_animation.set()
            if default_animation_thread and default_animation_thread.is_alive():
                default_animation_thread.join()

        # Pre-encode facial data before sending to Unreal Engine
        encoded_facial_data = pre_encode_facial_data(generated_facial_data, py_face, fps=60)
        run_audio_animation(audio_bytes, encoded_facial_data, py_face, create_socket_connection())

        preprocessed_data_queue.task_done()

        with queue_lock:
            if preprocessed_data_queue.empty() and request_queue.empty():
                stop_default_animation.clear()
                default_animation_thread = Thread(target=default_animation_loop, args=(py_face,))
                default_animation_thread.start()
