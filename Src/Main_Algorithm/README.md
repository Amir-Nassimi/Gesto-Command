# Introduction 
Action recognition using pose estimation is a computer vision task that involves identifying and classifying human actions based on analyzing the poses of the human body.

In this task, a deep learning model is trained to detect and track human body joints, such as the elbows, wrists, and knees, from video frames or images. Once the joints are located, the model estimates the pose of the person in the frame, which is a set of joint angles that define the configuration of the body.

Based on the pose, the model then identifies the action being performed by the person, such as walking, running, jumping, or dancing. The model can recognize various types of actions by analyzing the changes in the pose over time and classifying them using machine learning techniques.

Action recognition using pose estimation has a wide range of applications, including in sports analytics, human-robot interaction, and surveillance systems.

**Caution**: `This algorithm does not have the ability to identify multiple commands together, so it is recommended that a method for authentication be performed.`

# Installation
- Install ***Cuda Toolkit*** `11.8`.

- Install ***cuDNN*** `8.7`.

- Create python `3.10.14` env.

- Install the Pytorch necessary libs:

    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
    
- Install Tensorflow necessary lib:

    ```bash
    pip3 install tensorflow==2.14.0
    ```

- Navigate to the root directory of the Getso Command project and run the following commands to install necessary libraries:

    ```bash
    pip install -r requirements.txt
    ```

# Prediction

```bash
python ./Test/test.py
```

# Usage
```python
from Code.Offline_Pose import Pose

Pose().Pose_Command_(file_adrs=f'{os.path.abspath(Path(__file__).resolve().parents[3])}/Dataset/test.mp4', ROI_Region=[192,1,1015,747])
```