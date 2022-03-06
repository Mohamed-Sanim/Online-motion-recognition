# Online motion recognition
Human activities recognition is an important research focus area that has been the subject of a lot of research in the past two decades because it is so essential in many areas such as security, health, daily activity, robotics, and energy consumption.

In many applications, simply recognizing a single gesture is not enough, we may need to detect human movements moment by moment, especially in monitoring systems. Since in a previous code we built the model "SPD Siamese network" capable of recognizing hand gestures with high performance in segmented skeletal sequences, we decided to develop this model so that we could also identify human activities that depend on other parts of the body, not just the hands. Then, we build a system based on this model in order to detect different human activities in an unsegmented sequence.

Online action recognition is based on detecting the existence of an action as early as possible after its beginning and trying to determine the nature of this movement within a sequence of frames, without having information about the number of gestures present within the video, their nature or the length of time required to perform them or the start time, unlike the segmented action recognition. 

We use this model to build an online recognition system composed from a detector which segmented the action in a skeletal sequence and a classifier which identify the nature of the action or the gesture. 
