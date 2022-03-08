# Online motion recognition
Human activities recognition is an important research focus area that has been the subject of a lot of research in the past two decades because it is so essential in many areas such as security, health, daily activity, robotics, and energy consumption.

In many applications, simply recognizing a single gesture is not enough, we may need to detect human movements moment by moment, especially in monitoring systems. Since in a previous code we built the model "SPD Siamese network" capable of recognizing hand gestures with high performance in segmented skeletal sequences, we decided to develop this model so that we could also identify human activities that depend on other parts of the body, not just the hands. Then, we build a system based on this model in order to detect different human activities in an unsegmented sequence.

Online action recognition is based on detecting the existence of an action as early as possible after its beginning and trying to determine the nature of this movement within a sequence of frames, without having information about the number of gestures present within the video, their nature or the length of time required to perform them or the start time, unlike the segmented action recognition. 

We use this model to build an online recognition system composed from a detector which segmented the action in a skeletal sequence and a classifier which identify the nature of the action or the gesture. 

# How to build an online recognition system
We work with python 3.9.7 version. You need just about 4 or 5 lines of code to perform an online system execution. You need just to follow carefully the steps bellow.
## Clone the repository
To clone this github repository, use the command:
`!git clone https://github.com/Mohamed-Sanim/Online-motion-recognition.git`
Then, change the current working directory to this github repository using the command
```
cd Online-motion-recognition-main
#or choose the path in which the repository exists
cd <path-to-Online-motion-recognition-main-repository> 
```
## Installing libraries
All the libraries that you will use are listed in the file packages.txt. If there's just some libraries not installed, you can just install them one by on using: `pip install <name-of-library>`.  If you need to install all the required libraries, use the command `pip install –r packages.txt`
## How to run offline experiments
In this step,  you have to train the classifier and the detector in the offline state. For this purpose, you have to run offline_experiments.py after setting the convenient arguments. Before, check the following table of arguments of offline_experiments.py.

| Argument | Description | Requirement |
| --- | --- | --- |
| path | |  |
| dataset | |  |
| execution | |  |
| downloaded | |  |
| interpolation | |  |
| t0 | |  |
| NS | |  |
| eps | |  |
| outs_trans | |  |
| lr | |  |
| margin | |  |
| m | |  |
| ws | |  |
| learning_epochs | |  |
| classification_epochs | |  |
| feature | |  |


`!python offline_experiments.py --path <path-to-dataset>`
