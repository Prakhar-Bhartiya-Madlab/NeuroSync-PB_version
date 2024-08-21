# This code is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# For more details, visit: https://creativecommons.org/licenses/by-nc/4.0/

# audio_processing.py

from processing.audio_processing_utils import pad_audio_chunk, decode_audio_chunk, concatenate_outputs, ensure_2d  

def process_audio_features(audio_features, model, device, config):
    all_decoded_outputs = decode_audio(audio_features, model, device, config)
    final_decoded_outputs = postprocess_decoded_outputs(all_decoded_outputs)
    return final_decoded_outputs

def postprocess_decoded_outputs(final_decoded_outputs):
    final_decoded_outputs = ensure_2d(final_decoded_outputs)
    final_decoded_outputs[:, :61] /= 100 # scale the 0-100 down to 0 - 1 before playback.
    return final_decoded_outputs

def decode_audio(normalized_audio_features, model, device, config):
    frame_length = config['frame_size']
    num_features = normalized_audio_features.shape[1]
    num_frames = normalized_audio_features.shape[0]
    all_decoded_outputs = []

    model.eval()
    
    for start_idx in range(0, num_frames, frame_length):
        end_idx = min(start_idx + frame_length, num_frames)
        audio_chunk = normalized_audio_features[start_idx:end_idx]
        audio_chunk = pad_audio_chunk(audio_chunk, frame_length, num_features)
        decoded_outputs = decode_audio_chunk(audio_chunk, model, device)
        all_decoded_outputs.append(decoded_outputs[:end_idx - start_idx])

    return concatenate_outputs(all_decoded_outputs, num_frames)
