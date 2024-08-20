# NeuroSync
Audio to face blendshape inference helper code.

If using LiveLink local api, set your IP in :  livelink > connect > livelink_init.py 

Youtube demo : https://www.youtube.com/watch?v=ZgUVQkhiPi8 

Huggingface to get the model : https://huggingface.co/AnimaVR/NeuroSync-0.1a 

UE5 Demo project : https://drive.google.com/file/d/1jpG91_29ohro7YA3Mr6PF4S3TUVBuMGo/view?usp=sharing

A more permissive licence will come with later versions of the model.

Dimensions output are  :

EyeBlinkLeft	EyeLookDownLeft	EyeLookInLeft	EyeLookOutLeft	EyeLookUpLeft	EyeSquintLeft	EyeWideLeft	EyeBlinkRight	EyeLookDownRight	EyeLookInRight	EyeLookOutRight	EyeLookUpRight	EyeSquintRight	EyeWideRight	JawForward	JawRight	JawLeft	JawOpen	MouthClose	MouthFunnel	MouthPucker	MouthRight	MouthLeft	MouthSmileLeft	MouthSmileRight	MouthFrownLeft	MouthFrownRight	MouthDimpleLeft	MouthDimpleRight	MouthStretchLeft	MouthStretchRight	MouthRollLower	MouthRollUpper	MouthShrugLower	MouthShrugUpper	MouthPressLeft	MouthPressRight	MouthLowerDownLeft	MouthLowerDownRight	MouthUpperUpLeft	MouthUpperUpRight	BrowDownLeft	BrowDownRight	BrowInnerUp	BrowOuterUpLeft	BrowOuterUpRight	CheekPuff	CheekSquintLeft	CheekSquintRight	NoseSneerLeft	NoseSneerRight	**(52 to 61 should be ignored, for now) TongueOut	HeadYaw	HeadPitch	HeadRoll	LeftEyeYaw	LeftEyePitch	LeftEyeRoll	RightEyeYaw	RightEyePitch	RightEyeRoll (exclude these from sending to LiveLink**	| Angry	Disgusted	Fearful	Happy	Neutral	Sad	Surprised

![image](https://github.com/user-attachments/assets/f0e8063d-f03c-4a34-8f2b-e6f581b3f418)

The model processes batches of audio feature frames and outputs corresponding face blendshapes and emotion data.
