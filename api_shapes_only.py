# This code is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# For more details, visit: https://creativecommons.org/licenses/by-nc/4.0/

from flask import Flask, request, jsonify
import numpy as np
import torch
from processing.audio_processing import process_audio_features
from extraction.extract_features import extract_audio_features
from generate_face_shapes import generate_facial_data_from_bytes
from model import load_model

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

def preprocess_audio(audio_bytes):
    return generate_facial_data_from_bytes(audio_bytes, model, device)

audio_file_path = 'sample_data/audio.wav'
extracted_features, _ = extract_audio_features(audio_file_path)
final_decoded_outputs = process_audio_features(extracted_features, model, device, config)
print("Initial test output:", final_decoded_outputs)

@app.route('/audio_to_blendshapes', methods=['POST'])
def audio_to_blendshapes_route():
    audio_bytes = request.data
    
    generated_facial_data = preprocess_audio(audio_bytes)
    
    generated_facial_data_list = generated_facial_data.tolist() if isinstance(generated_facial_data, np.ndarray) else generated_facial_data
    
    return jsonify({'blendshapes': generated_facial_data_list})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7777)
