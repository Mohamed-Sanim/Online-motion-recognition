<img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1">
# Online motion recognition
Human activities recognition is an important research focus area that has been the subject of a lot of research in the past two decades because it is so essential in many areas such as security, health, daily activity, robotics, and energy consumption.

In many applications, simply recognizing a single gesture is not enough, we may need to detect human movements moment by moment, especially in monitoring systems. Since in a previous code we built the model "SPD Siamese network" capable of recognizing hand gestures with high performance in segmented skeletal sequences, we decided to develop this model so that we could also identify human activities that depend on other parts of the body, not just the hands. Then, we build a system based on this model in order to detect different human activities in an unsegmented sequence.

Online action recognition is based on detecting the existence of an action as early as possible after its beginning and trying to determine the nature of this movement within a sequence of frames, without having information about the number of gestures present within the video, their nature or the length of time required to perform them or the start time, unlike the segmented action recognition. 

We use this model to build an online recognition system composed from a detector which segmented the action in a skeletal sequence and a classifier which identify the nature of the action or the gesture. 
# Datasets
To evaluate our system, we choose to work with four dataset :
  - **Online Dynamic Hand Gesture dataset (ODHG)**: is a dataset consisting of 280 relatively long video clips ($\ge$600 frames) taken with the help of 28 volunteers, captured by "the Intel RealSense short range depth camera" (30 frames per second). The dataset contains sequences of 14 hand gestures.
  - **UOW Online Action 3D dataset (UOW)**: consists of skeleton sequences containing 48 sequences taken from 20 volunteers and  recorded using the kinect V2 sensor at with average 20 fps.
  - **Online Action Detection dataset (OAD)**: was captured using the kinect V2 sensor, which collects color images, depth images and human skeleton joints synchronously accompanying by the groudtruth. It gives the coordinates of 25 joints from different body parts. It includes 59 long sequences containing 104098 frames representing 839 actions captured at 8 fps.
  - **Industrial Human Action Recognition Datase (InHard)**: is a dataset collected in the context of human robot collaboration, with the help of 16 volunteers. It contains 38 long sequences that were captured by 'Combination Perception Neuron 32 Edition v2' motion sensor. The skeletal data comprises the 3D coordinates and the 3 rotation around the axis of 21 body joints 
  
Each dataset is accompanied by a description  nd folder named "Online" in which you finnd long sequences distributed into training sequences and test sequences. You find the groundtruth with each sequence. 

The directory is structured as follows:
  - Joints coordinates: \<path\>\/<name-of-dataset>/Online/<train/test>\/<name-of-sequence>\/skeletal_sequence.txt. Each line indicates the 3D coordinates of the captured joints(X_J1, Y_J1, Z_J1, X_J2, $Y_J2$, Z_J2, X_J3, Y_J3, Z_J3...)
  - Groundtruth information: \<path\>\/<name-of-dataset>\/Online/<train/test>/<name-of-sequence>/groundtruth.txt. Each file is an array of 3 columns. The first column correspond to the motion class and the two others indicate the time interval of each motion.
# How to build an online recognition system
We work with python 3.9.7 version. You need just about 4 or 5 lines of code to perform an online system execution. You need just to follow carefully the steps bellow.
## Clone the repository
To clone this github repository, use the command:
```
!git clone https://github.com/Mohamed-Sanim/Online-motion-recognition.git
```
Then, change the current working directory to this github repository using the command
```
cd Online-motion-recognition-main
#or choose the path in which the repository exists
cd <path-to-Online-motion-recognition-main-repository> 
```
## Installing libraries
All the libraries that you will use are listed in the file packages.txt. If there's just some libraries not installed, you can just install them one by on using: `pip install <name-of-library>`.  If you need to install all the required libraries, use the command `pip install –r packages.txt`
## How to  perform offline experiments
In this step,  you have to train the classifier and the detector in the offline state. For this purpose, you have to run offline_experiments.py after setting the convenient arguments. Before, check the following table of arguments of offline_experiments.py.

| Argument | Description |type | Default |Requirement |
| --- | --- | --- | --- | --- |
| path | specifies the directory path in which you install the dataset. This path must be written under the form "/../..../..", for example "C:/Users/ASUS/OneDrive/Desktop" | str | - | **Required** |
| dataset | specifies the dataset in which you perform experiments. We propose to work with four datasets \["OAD", "InHard", "ODHG", "UOW"\]. You have to write one of them| str| "OAD | Not required |
| execution | specifies if you would run a classifier or a detector. The possible values are "Classifier" and "Detector" | str | "Classifier" | Not required |
| interpolation | specifies the number of frames the skeletal sequences would be interpolated to| int | 500 | Not required |
| t0 | the time interval of execution in the first Gauss aggregation layer of ST studies  | int| 1 | Not required |
| NS | number of secondary subsequences in each primary subsequnece during the temporal - spatial studies | int | 15 | Not required |
| eps | threshold of the ReEig layer | float | 0.0001 | Not required|
| outs_trans | specifies the size of the SPD matrix output of SPD Aggregation layer | int | 200 | Not required |
| lr_learning | optimizer learning rate of the learning model | float| 1e-5 | Not required  |
| lr_classification | optimizer learning rate of the classification model | float| 7e-4 | Not required  |
| margin | specifies the margin of the contrastive loss function of the Siamese network| float | 7.0 | Not required |
| m | specifies the refresh rate of the detector | int | - |  **Required** if you execute a Detector |
| ws | specifies the window size of the detector (number of frames in each window | int | - | **Required** if you execute a Detector |
| learning_epochs | specfies the number of epoch needed for SPD learning | int | 10 | Not required |
| classification_epochs |specfies the number of epoch needed for SPD classification | int | 100 | Not required |

To run the classifier, you need just to specify the path in which the dataset exists or will be downloaded, and the name of the dataset. You can also modify the default values of the other arguments.
```
!python offline_experiments.py --path <path-to-dataset>  --dataset <name-of-the-dataset>
#Example 
!python offline_experiments.py --path C:/Users/ASUS/OneDrive/Desktop"   --dataset "InHard"
```

For the detector you need to specify in addition the execution type (since "detector" is not the default value), its refresh rate m and its window size ws. It is advised also to reduce the number of frames in the interploation. Here an example of execution.
```
!python offline_experiments.py --path C:/Users/ASUS/OneDrive/Desktop"   --dataset "ODHG"  --execution "Detector"  --m 6  --ws  24  --interpolation 100
```
## How to perform an online test
Once you perform the offline experiments on the Classifier and the Detector, you can perform the online experiments. You have just to specify the path, the dataset. You can also the number of tests in the verification process (set as default 3). Here an example of the execution of an online test.
```
!python online_experiments.py --path C:/Users/ASUS/OneDrive/Desktop"   --dataset "UOW"  --te 5
```
The output results of the experiment is a table describing the perforamnce of the model with respect to different metrics(Acccuracy, SL-score, EL-score, F1-score...).

**NB:** You can't run an online test without having run the classifier and the detector in a previous time.
