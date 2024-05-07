# In[1]:

# Normal Libs
import cv2
import mediapipe as mp
import numpy as np

# Detection Libs
import torch
from einops.layers.tensorflow import Rearrange
from singleton_decorator import singleton
from tensorflow.keras.layers import (
    GRU,
    Bidirectional,
    Conv1D,
    Dense,
    InputLayer,
    TimeDistributed,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import L1L2, L2

import os
from pathlib import Path

# In[2]:


@singleton
class Pose_Detection:
    def __init__(self):
        holistic_model = mp.solutions.holistic
        self.kp_model = holistic_model.Holistic(
            static_image_mode=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.5,
            model_complexity=2,
        )

    def KP_Detection(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False

        predict = self.kp_model.process(img)

        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img, predict

    def ExtractKeypoints(self, results):
        pose = (
            np.array(
                [[res.x, res.y] for res in results.pose_landmarks.landmark]
            ).flatten()
            if results.pose_landmarks
            else np.zeros(33 * 2)
        )
        lh = (
            np.array(
                [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
            ).flatten()
            if results.left_hand_landmarks
            else np.zeros(21 * 3)
        )
        rh = (
            np.array(
                [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
            ).flatten()
            if results.right_hand_landmarks
            else np.zeros(21 * 3)
        )

        return np.expand_dims(np.concatenate([pose, lh, rh]), axis=1)


# In[3]:

@singleton
class Person_Detection:
    def __init__(self, model_path = f'{os.path.abspath(Path(__file__).resolve().parents[3])}/Models/yolov5x6.pt'):
        self.pd_model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=model_path,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )

    def Person_Detection(self, img):
        points = []

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False

        with torch.no_grad():
            results = self.pd_model(img)

        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if len(results.xyxy[0]) == 0:
            return []

        for indx, res in enumerate(results.xyxy[0]):
            predict = results.pandas().xyxy[0]["name"][indx]

            if predict != "person":
                continue

            res = np.array(res.detach().cpu())
            if res[4] * 100 < 45:
                continue
            else:
                x = round(res[0])
                y = round(res[1])
                w = round(res[2])
                h = round(res[3])

                points.append((img[y:h, x:w], [x, y, h, w]))

        return points


# In[4]:


@singleton
class Pose_Command:
    def __init__(self, model_path=f'{os.path.abspath(Path(__file__).resolve().parents[3])}/Models/Model_Test_25_Epochs_Batch_64_Too_Deep_Categorical.h5'):
        self.thresh = 0.98
        self.no_seqs = 50  # No. of Videos per Action
        self.seqs_length = 30  # No. if Frames in Videos

        self.actions = np.array(["Play Music", "Hello", "Alarm 2"])

        self.model = Sequential(
            [
                InputLayer(input_shape=(self.seqs_length, 192, 1)),
                TimeDistributed(
                    Conv1D(
                        1,
                        1,
                        activation="relu",
                        kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
                        bias_regularizer=L2(1e-4),
                    )
                ),
                Rearrange("b s d t -> b s (d t)"),
                Bidirectional(
                    GRU(
                        64,
                        return_sequences=True,
                        activation="relu",
                        input_shape=(30, 128),
                        kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
                        bias_regularizer=L2(1e-4),
                    )
                ),
                Bidirectional(
                    GRU(
                        128,
                        return_sequences=True,
                        activation="relu",
                        input_shape=(30, 128),
                        kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
                        bias_regularizer=L2(1e-4),
                    )
                ),
                Bidirectional(
                    GRU(
                        64,
                        return_sequences=False,
                        activation="relu",
                        kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
                        bias_regularizer=L2(1e-4),
                    )
                ),
                Dense(
                    64,
                    activation="relu",
                    kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
                    bias_regularizer=L2(1e-4),
                ),
                Dense(
                    128,
                    activation="relu",
                    kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
                    bias_regularizer=L2(1e-4),
                ),
                Dense(
                    32,
                    activation="relu",
                    kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
                    bias_regularizer=L2(1e-4),
                ),
                Dense(
                    self.actions.shape[0],
                    activation="softmax",
                    kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
                    bias_regularizer=L2(1e-4),
                ),
            ]
        )

        self.Load_Model(model_path)

    def Warm_Up(self):
        print("Warming Up Model")
        self.model.predict(np.zeros((1, 30, 192, 1)))
        print("Model Warmed up !!")

    def Load_Model(self,path):
        self.Warm_Up()
        self.model.load_weights(path)
        print("Pretrained weights loaded !!")


# In[5]:


class Execute:
    def __init__(self):
        self.pose_command = Pose_Command()
        self.pose_detection = Pose_Detection()
        self.person_detection = Person_Detection()

    def Execute_Sequences(self, img, sequences):
        points = self.person_detection.Person_Detection(img.copy())

        if len(points) == 0:
            return False, sequences, []

        for img, (x, y, h, w) in points:
            _, results = self.pose_detection.KP_Detection(img)

            key_points = self.pose_detection.ExtractKeypoints(results)

            sequences.append(key_points.astype("float32"))

        return True, sequences, [x, y, h, w]

    def Execute_Command(self, sequence):
        poses_seq = np.expand_dims(sequence, axis=0)
        res = self.pose_command.model.predict(poses_seq)[0]
        return res


# In[6]:

class Pose:
    @staticmethod
    def Pose_Command_(
        file_adrs,
        ROI_Region=[192, 1, 1015, 747],
    ):
        counter = 0
        execution_obj = Execute()
        sequences, predictions_dict = [], {}
        seq_len = execution_obj.pose_command.seqs_length

        if "rtsp" in file_adrs:
            real_time = True
        else:
            real_time = False

        stream = cv2.VideoCapture(file_adrs)

        while stream.isOpened():
            _, img_ = stream.read()

            if img_ is None:
                img_ = np.zeros([750, 1250, 3], dtype=np.uint8)
            else:
                img_ = cv2.resize(np.array(img_), (1250, 750))

            try:
                if ROI_Region[0] == ROI_Region[1] and ROI_Region[2] == ROI_Region[3]:
                    if ROI_Region[0] == 0 and ROI_Region[0] == ROI_Region[3]:
                        pass
                    else:
                        img = img_[
                            int(ROI_Region[1]) : int(ROI_Region[1] + ROI_Region[3]),
                            int(ROI_Region[0]) : int(ROI_Region[0] + ROI_Region[2]),
                        ]
                else:
                    img = img_[
                        int(ROI_Region[1]) : int(ROI_Region[1] + ROI_Region[3]),
                        int(ROI_Region[0]) : int(ROI_Region[0] + ROI_Region[2]),
                    ]

            except IndexError:
                pass

            flag, sequences, cordinations = execution_obj.Execute_Sequences(img, sequences)

            if not flag:
                continue

            counter += 1

            if len(sequences[-seq_len:]) == seq_len:
                predictions = execution_obj.Execute_Command(sequences[-seq_len:])

                results = {
                    "actions": execution_obj.pose_command.actions,
                    "scores": predictions,
                    "predict": "Unknown"
                    if np.max(predictions) < execution_obj.pose_command.thresh
                    else execution_obj.pose_command.actions[np.argmax(predictions)],
                    "location": cordinations,
                }

                if not real_time:
                    predictions_dict[f"{counter}"] = results
                else: pass
                    # insert_data(results)  # Khode Hesam ezafe konad

                print(
                    f"Actions are: {execution_obj.pose_command.actions} - Predictions are: {predictions} - Final Pred is : {'Unknown' if np.max(predictions) < execution_obj.pose_command.thresh else execution_obj.pose_command.actions[np.argmax(predictions)]}"
                )

            else:
                continue

        stream.release()

        return predictions_dict


# In[7]:

# Pose().Pose_Command_()
