# This code is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# For more details, visit: https://creativecommons.org/licenses/by-nc/4.0/


# extract_features.py

import numpy as np
import librosa
import io

from extraction.extract_features_utils import extract_overlapping_mfcc, reduce_features

def load_and_preprocess_audio(audio_path, sr=88200):
    y, sr = load_audio(audio_path, sr)
    if sr != 88200:
        y = librosa.resample(y, orig_sr=sr, target_sr=88200)
        sr = 88200
    
    return y, sr

def load_audio(audio_path, sr=88200):
    y, sr = librosa.load(audio_path, sr=sr)
    print(f"Loaded audio file '{audio_path}' with sample rate {sr}")
    return y, sr

def load_audio_from_bytes(audio_bytes, sr):
    audio_file = io.BytesIO(audio_bytes)
    y, sr = librosa.load(audio_file, sr=sr)
    return y, sr


def extract_audio_features(audio_input, sr=88200, from_bytes=False):
    if from_bytes:
        y, sr = load_audio_from_bytes(audio_input, sr)
    else:
        y, sr = load_and_preprocess_audio(audio_input, sr)
    
    frame_length = int(0.01667 * sr)  # Frame length set to 0.01667 seconds (~60 fps)
    hop_length = frame_length // 2  # 2x overlap for smoother transitions
    min_frames = 9  # Minimum number of frames needed for delta calculation

    num_frames = (len(y) - frame_length) // hop_length + 1

    if num_frames < min_frames:
        print(f"Audio file is too short: {num_frames} frames, required: {min_frames} frames")
        return None, None

    combined_features = extract_and_combine_features(y, sr, frame_length, hop_length)
    
    return combined_features, y

def extract_and_combine_features(y, sr, frame_length, hop_length):
    all_features = []
    num_mfcc = 26 
    
    mfcc_features = extract_overlapping_mfcc(y, sr, num_mfcc, frame_length, hop_length)
    reduced_mfcc_features = reduce_features(mfcc_features)
   
    all_features.append(reduced_mfcc_features.T)  
   
    combined_features = np.hstack(all_features)
    
    return combined_features
