# This code is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# For more details, visit: https://creativecommons.org/licenses/by-nc/4.0/

import pygame
from flask import Flask, request, jsonify
from threading import Thread
from queue import Queue, Empty
import torch 

from model import load_model

from utils.api_utils import initialize_directories, process_preprocessing_queue, process_playback_queue, queue_lock

from livelink.animations.default_animation import default_animation_loop, stop_default_animation
from livelink.connect.livelink_init import create_socket_connection, initialize_py_face

config = {
    'sr': 88200,  
    'frame_rate': 60,           
    'hidden_dim':  1024,   
    'n_layers': 4,
    'num_heads': 4,    
    'dropout': 0.0,        
    'output_dim': 68,      
    'input_dim': 26 + 26 + 26, 
    'frame_size': 256,
}

model_path = '_out/model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model(model_path, config, device)


app = Flask(__name__)

py_face = initialize_py_face()
socket_connection = create_socket_connection()

default_animation_thread = Thread(target=default_animation_loop, args=(py_face,))
default_animation_thread.start()

request_queue = Queue()
preprocessed_data_queue = Queue()

initialize_directories()

@app.route('/audio_to_face', methods=['POST'])
def play_audio_route():
    audio_bytes = request.data

    with queue_lock:
        request_queue.put(audio_bytes)
    
    return jsonify({'status': 'queued'})

@app.route('/clear_queue', methods=['POST'])
def clear_queue_route():
    global stop_default_animation, default_animation_thread

    with queue_lock:
        while not request_queue.empty():
            try:
                request_queue.get_nowait()
            except Empty:
                break

        while not preprocessed_data_queue.empty():
            try:
                preprocessed_data_queue.get_nowait()
            except Empty:
                break

    return jsonify({'status': 'cleared'})

if __name__ == '__main__':
    preprocessing_queue_thread = Thread(target=process_preprocessing_queue, args=(request_queue, preprocessed_data_queue, model, device))
    preprocessing_queue_thread.start()
    
    playback_queue_thread = Thread(target=process_playback_queue, args=(preprocessed_data_queue, py_face, default_animation_thread, request_queue))
    playback_queue_thread.start()
    
    try:
        app.run(host='0.0.0.0', port=7777)
    finally:
        stop_default_animation.set()
        if default_animation_thread:
            default_animation_thread.join()
        pygame.quit()
        
        with queue_lock:
            request_queue.put(None)
        preprocessing_queue_thread.join()

        preprocessed_data_queue.put((None, None))
        playback_queue_thread.join()