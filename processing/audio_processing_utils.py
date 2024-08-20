# This code is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# For more details, visit: https://creativecommons.org/licenses/by-nc/4.0/


# audio_processing_utils.py

import numpy as np
import torch

def pad_audio_chunk(audio_chunk, frame_length, num_features):
    if audio_chunk.shape[0] < frame_length:
        pad_length = frame_length - audio_chunk.shape[0]
        padding = np.pad(
            audio_chunk,
            pad_width=((0, pad_length), (0, 0)),
            mode='reflect'
        )
        audio_chunk = np.vstack((audio_chunk, padding[-pad_length:, :num_features]))
    return audio_chunk

def decode_audio_chunk(audio_chunk, model, device):
    src_tensor = torch.tensor(audio_chunk, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        encoder_outputs = model.encoder(src_tensor)
        output_sequence = model.decoder(encoder_outputs)
        decoded_outputs = output_sequence.squeeze(0).cpu().numpy()
    return decoded_outputs

def concatenate_outputs(all_decoded_outputs, num_frames):
    final_decoded_outputs = np.concatenate(all_decoded_outputs, axis=0)
    final_decoded_outputs = final_decoded_outputs[:num_frames]
    return final_decoded_outputs

def ensure_2d(final_decoded_outputs):
    if final_decoded_outputs.ndim == 3:
        final_decoded_outputs = final_decoded_outputs.reshape(-1, final_decoded_outputs.shape[-1])
    return final_decoded_outputs
