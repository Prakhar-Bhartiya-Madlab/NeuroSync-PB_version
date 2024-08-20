# This code is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# For more details, visit: https://creativecommons.org/licenses/by-nc/4.0/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def pad_data(df1, df2):
    max_len = max(len(df1), len(df2))
    if len(df1) < max_len:
        pad_len = max_len - len(df1)
        padding = pd.DataFrame(0, index=np.arange(pad_len), columns=df1.columns)
        df1 = pd.concat([df1, padding], ignore_index=True)
    elif len(df2) < max_len:
        pad_len = max_len - len(df2)
        padding = pd.DataFrame(0, index=np.arange(pad_len), columns=df2.columns)
        df2 = pd.concat([df2, padding], ignore_index=True)
    return df1, df2

def plot_comparison(ground_truth_path, generated_path, output_image_path):
    ground_truth = pd.read_csv(ground_truth_path)
    generated = pd.read_csv(generated_path)

    ground_truth, generated = pad_data(ground_truth, generated)

    # Select the first 512 frames
    ground_truth = ground_truth.head(512)
    generated = generated.head(512)

    timecodes = ground_truth['Timecode'].astype(str)

    features_to_plot = ['JawOpen']

    plt.figure(figsize=(20, 20))
    for feature in features_to_plot:
        plt.plot(timecodes, ground_truth[feature], label=f'Ground Truth {feature}')
        plt.plot(timecodes, generated[feature], label=f'Generated {feature}', linestyle='dashed')

    plt.legend()
    plt.xticks(rotation=45)
    plt.xlabel('Timecode')
    plt.ylabel('Feature Value')
    plt.title('Comparison of Ground Truth and Generated Facial Features')
    plt.tight_layout()

    plt.savefig(output_image_path, dpi=100)
    plt.close()
    print(f"Comparison plot saved to {output_image_path}")
