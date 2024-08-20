# NeuroSync
Audio to face blendshape inference helper code.

Huggingface to get the model : https://huggingface.co/AnimaVR/NeuroSync-0.1a 

This guide will help you convert audio input into real-time face animation data using our alpha version foundation model.

A more permissive licence will come with later versions of the model.

Dimensions output are (52 to 61 should be ignored, for now) :

EyeBlinkLeft	EyeLookDownLeft	EyeLookInLeft	EyeLookOutLeft	EyeLookUpLeft	EyeSquintLeft	EyeWideLeft	EyeBlinkRight	EyeLookDownRight	EyeLookInRight	EyeLookOutRight	EyeLookUpRight	EyeSquintRight	EyeWideRight	JawForward	JawRight	JawLeft	JawOpen	MouthClose	MouthFunnel	MouthPucker	MouthRight	MouthLeft	MouthSmileLeft	MouthSmileRight	MouthFrownLeft	MouthFrownRight	MouthDimpleLeft	MouthDimpleRight	MouthStretchLeft	MouthStretchRight	MouthRollLower	MouthRollUpper	MouthShrugLower	MouthShrugUpper	MouthPressLeft	MouthPressRight	MouthLowerDownLeft	MouthLowerDownRight	MouthUpperUpLeft	MouthUpperUpRight	BrowDownLeft	BrowDownRight	BrowInnerUp	BrowOuterUpLeft	BrowOuterUpRight	CheekPuff	CheekSquintLeft	CheekSquintRight	NoseSneerLeft	NoseSneerRight	**TongueOut	HeadYaw	HeadPitch	HeadRoll	LeftEyeYaw	LeftEyePitch	LeftEyeRoll	RightEyeYaw	RightEyePitch	RightEyeRoll**	| Angry	Disgusted	Fearful	Happy	Neutral	Sad	Surprised

![image](https://github.com/user-attachments/assets/f0e8063d-f03c-4a34-8f2b-e6f581b3f418)


The model processes batches of audio feature frames and outputs corresponding face blendshapes and emotion data.

Input Requirements

Audio Format: Your input can be either:

A raw byte stream, or
A .wav file

Note: Regardless of the input format, the audio must be upsampled to a sample rate of 88,200 Hz before feature extraction. This ensures high accuracy when processing overlapping frames.

Preparing Your Input Audio

Step 1: Load and Upsample the Audio

Load the Audio:

Load and upsample the audio to mono 88,200 Hz.

Step 2: Frame Extraction

Extract Overlapping Frames:

Split the upsampled audio into overlapping frames. The frame length is set to 0.01667 seconds (about 60 frames per second), which corresponds to a frame length of approximately 1470 samples at 88,200 Hz.

The hop length (the step between consecutive frames) is set to half the frame length, ensuring 50% overlap for smoother transitions between frames.

Check Frame Count:

Ensure the audio has enough frames for accurate processing. A minimum of 9 frames is required to compute the delta (first derivative) and delta-delta (second derivative) features.

Step 3: MFCC Extraction and Processing

Compute MFCCs:

For each frame, extract 26 MFCCs using the extract_overlapping_mfcc function.Include deltas and apply cepstral mean variance normalization.

Reduce and Smooth Features:

Reduce the dimensionality of the features by pairing and averaging adjacent frames.

Optionally, apply smoothing to the reduced features to further smooth out transitions between frames.

Step 4: Organize Into Batches

Batching:

Organize the processed frames into batches of 128 frames each. If your audio is longer, create multiple batches. Each batch will be processed individually by the model.

Feeding the Model

With your audio processed and organized into batches:

Input: Each batch contains 128 frames, each frame having 78 features derived from MFCCs and their deltas.

Model Processing: The model will process each batch and output the corresponding face animation frames and emotion data.

Model Output

For each batch of 128 frames:

Blendshapes: The model outputs 128 frames of face animation data, with 61 blendshapes per frame:

Use the first 51 blendshapes for controlling the face.

Ignore the remaining 10 blendshapes (52-61), as they are intended for neck movement and are processed by a different model.

Emotions: The model also outputs 7 emotion dimensions per frame (62 - 68 dimensions), which can be used to adjust the blendshapes for more expressive animation.

Encode-Decode Processing

Step 1: Encode the Audio Features

Prepare Audio Features:

The features extracted from the audio are encoded by the model. If the audio chunk is shorter than 128 frames, it will be padded to match the required frame length.

Decode the Encoded Features:

The model's decoder processes the encoded features and outputs the corresponding face blendshapes and emotion data.

Step 2: Post-Processing

Concatenate Frame Outputs:

After processing each batch, concatenate the output blendshape frames in sequence to form the complete face animation.

Apply Emotion Data:

Use the emotion dimensions from the model's output to automatically adjust the blendshapes, enhancing expressiveness based on the emotional content of each frame.

Optional Smoothing:

You can smooth the final animation by averaging adjacent frames, which will further refine the transitions.

Real-Time Processing Workflow

Process Entire Audio:

Start with a full audio clip, upsample it to 88,200 Hz, and break it down into overlapping frames.

Batch Inference:

Feed these frames to the model in batches of 128 frames. The model processes each batch individually and returns the corresponding face animation frames.

Reassemble:

As you receive output for each batch, concatenate the results to form a continuous animation sequence.

Return Final Animation:

The final result is a sequence of face animation frames corresponding to the input audio. This entire process is optimized to handle approximately 10 seconds of audio and return the corresponding face animation within 100 milliseconds, making it suitable for real-time applications.

Example Code

Here is how you would typically call the functions to generate facial data from audio bytes:

audio_features = extract_audio_features(audio_bytes)

final_decoded_outputs = process_audio_features(audio_features, model, device)

return final_decoded_outputs

This code provides the essential steps to convert an audio stream into face animation data, which includes encoding, decoding, and post-processing.
