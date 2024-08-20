
# This code is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# For more details, visit: https://creativecommons.org/licenses/by-nc/4.0/

import numpy as np
import pandas as pd

def save_generated_data_as_csv(generated, output_path):
    columns = [
        'Timecode', 'BlendshapeCount', 'EyeBlinkLeft', 'EyeLookDownLeft', 'EyeLookInLeft', 'EyeLookOutLeft', 'EyeLookUpLeft', 
        'EyeSquintLeft', 'EyeWideLeft', 'EyeBlinkRight', 'EyeLookDownRight', 'EyeLookInRight', 'EyeLookOutRight', 'EyeLookUpRight', 
        'EyeSquintRight', 'EyeWideRight', 'JawForward', 'JawRight', 'JawLeft', 'JawOpen', 'MouthClose', 'MouthFunnel', 'MouthPucker', 
        'MouthRight', 'MouthLeft', 'MouthSmileLeft', 'MouthSmileRight', 'MouthFrownLeft', 'MouthFrownRight', 'MouthDimpleLeft', 
        'MouthDimpleRight', 'MouthStretchLeft', 'MouthStretchRight', 'MouthRollLower', 'MouthRollUpper', 'MouthShrugLower', 
        'MouthShrugUpper', 'MouthPressLeft', 'MouthPressRight', 'MouthLowerDownLeft', 'MouthLowerDownRight', 'MouthUpperUpLeft', 
        'MouthUpperUpRight', 'BrowDownLeft', 'BrowDownRight', 'BrowInnerUp', 'BrowOuterUpLeft', 'BrowOuterUpRight', 'CheekPuff', 
        'CheekSquintLeft', 'CheekSquintRight', 'NoseSneerLeft', 'NoseSneerRight', 'TongueOut', 'HeadYaw', 'HeadPitch', 'HeadRoll', 
    'LeftEyeYaw', 'LeftEyePitch', 'LeftEyeRoll', 'RightEyeYaw', 'RightEyePitch', 'RightEyeRoll', 'Angry',  'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised', 
    ]
    
    # Reshape the generated array to match the expected dimensions
    generated = generated.reshape(-1, 68)
    
    # Generate timecodes based on the number of frames
    frame_count = generated.shape[0]
    timecodes = np.arange(frame_count)

    print(f"Generated shape: {generated.shape}")
    print(f"Timecodes shape: {timecodes.shape}")
    
    # Stack the timecodes and generated data together
    data = np.column_stack((timecodes, np.full((generated.shape[0], 1), generated.shape[1]), generated))
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_path, index=False)
    print(f"Generated data saved to {output_path}")
