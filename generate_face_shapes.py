# This code is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# For more details, visit: https://creativecommons.org/licenses/by-nc/4.0/

# generate_face_shapes.py

import numpy as np

from extraction.extract_features import extract_audio_features
from processing.audio_processing import process_audio_features

def generate_facial_data_from_bytes(audio_bytes, model, device, config, use_smoothing=True): # enable smoothing to reduce any stutter when increasing the scale in livelink > connect > pylivelinkface.py, more data/training will remove the need for this in the future.
    
    audio_features, y = extract_audio_features(audio_bytes, from_bytes=True)
    
    if audio_features is None or y is None:
        return [], np.array([])
  
    final_decoded_outputs = process_audio_features(audio_features, model, device, config)

    if use_smoothing: # this essentially takes a 60fps stuttery feed and blends frame pairs to smooth it if needed. Might be needed if scale is too high (anything over 1.2ish).
        final_decoded_outputs = smooth_by_averaging_pairs(final_decoded_outputs)

    return final_decoded_outputs

def smooth_by_averaging_pairs(data):
    smoothed_data = data.copy()
    for i in range(1, len(data)):
        smoothed_data[i - 1] = (data[i - 1] + data[i]) / 2.0
    return smoothed_data
