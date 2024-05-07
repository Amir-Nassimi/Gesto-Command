import os, sys
from pathlib import Path

sys.path.append(os.path.abspath(Path(__file__).resolve().parents[1]))
from Code.Offline_Pose import Pose

Pose().Pose_Command_(file_adrs=f'{os.path.abspath(Path(__file__).resolve().parents[3])}/Dataset/test.mp4', ROI_Region=[192,1,1015,747])
