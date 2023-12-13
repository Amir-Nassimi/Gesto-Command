import cv2
import numpy as np
import mediapipe as mp
from singleton_decorator import singleton


@singleton
class PoseDetection:
    def __init__(self):
        holistic_model = mp.solutions.holistic
        self.kp_model = holistic_model.Holistic(
            static_image_mode=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.5,
            model_complexity=1,
        )
        self.extract_keypoint_result = None

    def kp_detection(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False

        predict = self.kp_model.process(img)

        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img, predict

    def extract_keypoint(self, results):
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

        self.extract_keypoint_result = np.expand_dims(np.concatenate([pose, lh, rh]), axis=1)

        return self.extract_keypoint_result
