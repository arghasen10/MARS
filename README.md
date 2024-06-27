# MARS: Multi-user Activity tracking via Room-scale Sensing

Developing a pervasive interactive smart space requires continuous detection of human presence and their activities. Existing literature lacks robust wireless sensing mechanisms capable of continuously monitoring multiple users’ activities without prior knowledge of the environment. Developing such a mechanism requires simultaneous localization and tracking of multiple subjects. In addition, it requires identifying their activities at various scales, some being macro-scale activities like walking, lunges, squats, etc., while others are micro-scale activities like typing or playing guitar, etc. In this paper, we develop a holistic system called <b><i>MARS</i></b> using a single Commercial off-the-shelf (COTS) MillimeterWave (mmWave) radar, which employs an intelligent model to sense both macro and micro activities. In addition, it uses a dynamic spatial time-sharing approach to sense different subjects simultaneously at a room-scale. A thorough evaluation of MARS shows that it can sense up to 5 subjects overan 8 × 5 m<sup>2</sup> space with an average accuracy of 98% and 94% while identifying macro and micro activities, respectively.

More details on the experimental setup for different room experiments R1:office-cabin(4mx3m), R2:study-room(5mx8m) and R3:laboratory(12mx6.5m), different obstacles wooden-chairs, tables, metal-wardrobes, fiber-desks, desktop-computers, etc., users performing macro-micro activities together etc.is provided [here](./activity_classifier/evaluation/more_experiments/README.md).



## Data Collection Setup

To install mmWave Demo Visualizer from Texas Instruments, first go to this [link](https://dev.ti.com/gallery/view/mmwave/mmWave_Demo_Visualizer/) and select SDK [2.1.0](https://dev.ti.com/gallery/view/mmwave/mmWave_Demo_Visualizer/ver/2.1.0/). Now go to Help and select Download or Clone Visualizer. Finally you need to download and install the entire repository in your machine.

Now copy all the content of the provided submodule `mmWave-Demo-Visualizer` and paste it in the installaed mmWave-demo-visualizer directory i.e. **C:\Users\UserName\guicomposer\runtime\gcruntime.v11\mmWave_Demo_Visualizer**

Once you are done with the installation run 
```bash
launcher.exe
```
Finally using this tool you can save mmWave data in your local machine. Data will be saved in a txt file in JSON format.

## Feature Description

| **Feature**     | **Description**                                                                                                                                            |
| :-------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------- |
| datetime        | The date and time when the data was recorded. This helps in time-series analysis and synchronization with other data sources.                        |
| rangeIdx        | Index corresponding to the range bin of the detected object. It indicates the distance of the  object from the radar.                                 |
| dopplerIdx      | Index corresponding to the Doppler bin, which represents the relative velocity of the detected object.                                               |
| numDetectedObj  | The number of objects detected in a single frame. This feature is useful for understanding  the multi-user activity dynamics of the scene.            |
| range           | The actual distance measurement of the detected object from the radar in meters.                                                                           |
| peakVal         | The peak value of the detected signal, indicating the strength of the returned radar signal.                                                               |
| x\_coord        | The x-coordinate of the detected object in the radar's coordinate system.                                                                                  |
| y\_coord        | The y-coordinate of the detected object in the radar's coordinate system.                                                                                  |
| doppz           | The Range-doppler Heatmap value indicating the radial velocity of the detected object, helping to distinguish between stationary and moving objects. |
| Position        | The position of the subject with respect to the radar. It can have values like 2m, 3m and 5m.                                                              |
| Orientation     | The orientation of the subject relative to the radar's bore-sight angle: left, right, front, and back.                                                     |
| activity        | The specific activity being performed by the subject, such as walking, running, or typing, used for machine learning and classification tasks.       |
| activity\_class | A broad categorical label of the type of activity: whether macro activity  or micro activity                                                               |


## License and Ethical Approval

The codebase and dataset is free to download and can be used with GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007 for non-commercial purposes. All participants signed forms consenting to the use of collected activity sensing data for non-commercial research purposes. The Institute's Ethical Review Board (IRB) at IIT Kharagpur, India has approved the data collection under the study title: <b>"Human Activity Monitoring in Pervasive Sensing Setupfield"</b>, with the Approval Number: <b>IIT/SRIC/DEAN/2023, dated July 31, 2023</b>. Moreover, we have made significant efforts to anonymize the participants to preserve privacy while providing the useful and necessary information to encourage future research with the dataset.


## Reference
To refer <i>MARS</i> framework or the dataset, please cite the following work.

BibTex Reference:
```
@inproceedings{sen2024mars, 
title={Continuous Multi-user Activity Tracking via Room-Scale mmWave Sensing}, 
author={Sen, Argha and Das, Anirban and Pradhan, Swadhin and Chakraborty, Sandip},
booktitle={23rd ACM/IEEE Conference on Information Processing in Sensor Networks},
year={2024},
organization={IEEE}
} 
```

For questions and general feedback, contact Argha Sen (arghasen10@gmail.com).