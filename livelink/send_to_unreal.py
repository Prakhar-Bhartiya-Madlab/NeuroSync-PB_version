# This code is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# For more details, visit: https://creativecommons.org/licenses/by-nc/4.0/

# send_to_unreal.py

import time 
import numpy as np
from typing import List

from livelink.connect.livelink_init import create_socket_connection, FaceBlendShape
from livelink.animations.default_animation import default_animation_data

def pre_encode_facial_data(facial_data: List[np.ndarray], py_face, fps: int = 60) -> List[bytes]:
    """
    Pre-encodes facial data to reduce runtime encoding delays, including blend-in and blend-out effects.
    
    Args:
        facial_data (List[np.ndarray]): The facial data to encode.
        py_face: Instance of PyLiveLinkFace used to encode the data.
        fps (int): Frames per second to determine the blend-in and blend-out duration.

    Returns:
        List[bytes]: List of pre-encoded facial data frames.
    """
    encoded_data = []

    # Determine blend-in and blend-out frame counts
    blend_in_frames = int(0.1 * fps)
    blend_out_frames = int(0.3 * fps)

    # Apply blend-in
    for frame_index in range(blend_in_frames):
        weight = frame_index / blend_in_frames
        apply_blendshapes(facial_data[frame_index], weight, py_face)
        encoded_data.append(py_face.encode())

    # Encode the main animation frames
    for frame_data in facial_data[blend_in_frames:-blend_out_frames]:
        for i in range(min(len(frame_data), 51)):  # Ensure we only process the first 51 blendshapes
            py_face.set_blendshape(FaceBlendShape(i), frame_data[i])
        encoded_data.append(py_face.encode())

    # Apply blend-out
    for frame_index in range(blend_out_frames):
        weight = frame_index / blend_out_frames
        reverse_index = len(facial_data) - blend_out_frames + frame_index
        apply_blendshapes(facial_data[reverse_index], 1.0 - weight, py_face)
        encoded_data.append(py_face.encode())

    return encoded_data

def apply_blendshapes(frame_data: np.ndarray, weight: float, py_face):
    for i in range(51):  # Apply the first 51 blendshapes (no neck at the moment)
        default_value = default_animation_data[0][i]
        facial_value = frame_data[i]
        blended_value = (1 - weight) * default_value + weight * facial_value
        py_face.set_blendshape(FaceBlendShape(i), float(blended_value))

    # Handle new emotion dimensions (61 to 67)
    additional_values = frame_data[61:68]
    values_str = " ".join([f"{i+61}: {value:.2f}" for i, value in enumerate(additional_values)])
    print(f"Frame Values: {values_str}")

    # Determine the emotion with the highest value
    max_emotion_index = np.argmax(additional_values)
    emotions = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]
    print(f"Highest emotion: {emotions[max_emotion_index]} with value: {additional_values[max_emotion_index]:.2f}")

def send_pre_encoded_data_to_unreal(encoded_facial_data: List[bytes], start_event, fps: int, socket_connection=None):
    """
    Sends pre-encoded facial data to Unreal Engine, synchronizing it with the audio.

    Args:
        encoded_facial_data (List[bytes]): List of pre-encoded facial data frames.
        start_event: Event to synchronize the start with the audio.
        fps (int): Frames per second to control the playback speed.
        socket_connection: Socket connection to send the data to Unreal Engine.
    """
    try:
        own_socket = False
        if socket_connection is None:
            socket_connection = create_socket_connection()
            own_socket = True

        start_event.wait()  # Wait until it's time to start playback

        for frame_data in encoded_facial_data:
            socket_connection.sendall(frame_data)
            time.sleep(1 / fps)

    except KeyboardInterrupt:
        pass
    finally:
        if own_socket:
            socket_connection.close()
