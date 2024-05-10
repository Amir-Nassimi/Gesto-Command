import sys
import cv2
import argparse
import numpy as np

from load_stream import Camera
from pose_command import PoseCommand
from pose_detection import PoseDetection
from person_detection import Person_Detection


class Execute:
    def __init__(self, address):
        self.video_capt = Camera(address)
        self.pose_command = PoseCommand()
        self.pose_detection = PoseDetection()
        self.person_detection = Person_Detection()

    def execute_sequences(self, img, sequences):
        points = self.person_detection.Person_Detection(img.copy())

        if len(points) == 0:
            return False, sequences, []

        x, y, h, w = None, None, None, None

        for img, (x, y, h, w) in points:
            _, results = self.pose_detection.kp_detection(img)

            key_points = self.pose_detection.extract_keypoint(results)

            sequences.append(key_points.astype("float32"))

        return True, sequences[-self.pose_command.seq_len:], [x, y, h, w]

    def execute_command(self, sequence):
        poses_seq = np.expand_dims(sequence, axis=0)
        res = self.pose_command.model.predict(poses_seq)[0]
        return res


def pose_command_(file_address, roi_region):
    counter = 0
    execution_obj = Execute(file_address)
    sequences, predictions_dict = [], {}
    seq_len = execution_obj.pose_command.seqs_length

    execution_obj.video_capt.run()

    while not execution_obj.video_capt.exit_signal.is_set():
        img_ = execution_obj.video_capt.frames_queue.get()
        img_ = cv2.resize(np.array(img_), (1250, 750))

        img = None

        try:
            if roi_region[0] == roi_region[1] and roi_region[2] == roi_region[3]:
                if roi_region[0] == 0 and roi_region[0] == roi_region[3]:
                    pass
                else:
                    img = img_[
                          int(roi_region[1]): int(roi_region[1] + roi_region[3]),
                          int(roi_region[0]): int(roi_region[0] + roi_region[2]),
                          ]
            else:
                img = img_[
                      int(roi_region[1]): int(roi_region[1] + roi_region[3]),
                      int(roi_region[0]): int(roi_region[0] + roi_region[2]),
                      ]

        except IndexError:
            pass

        flag, sequences, coordinate = execution_obj.execute_sequences(img, sequences)

        if not flag:
            continue

        counter += 1

        if len(sequences) == seq_len:
            predictions = execution_obj.execute_command(sequences)

            results = {
                "actions": execution_obj.pose_command.actions,
                "scores": predictions,
                "predict": "Unknown"
                if np.max(predictions) < execution_obj.pose_command.thresh
                else execution_obj.pose_command.actions[np.argmax(predictions)],
                "location": coordinate,
            }

            predictions_dict[f"{counter}"] = results

            print(
                f"Actions are: {execution_obj.pose_command.actions} - Predictions are: {predictions} - Final Pred is : {'Unknown' if np.max(predictions) < execution_obj.pose_command.thresh else execution_obj.pose_command.actions[np.argmax(predictions)]}"
            )

        else:
            continue

    execution_obj.video_capt.stop_threads()

    return predictions_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gesto Command")
    
    parser.add_argument('-f',
                        '--filepath', help="File path of video or rtsp adrs of camera",
                        required=True)
    
    parser.add_argument(
        '-r', '--roi',
        help="Region Of Interest - default: No Roi would be set",
        default=[-1,-1,-1,-1],
        required=False)
    
    args = parser.parse_args()
    
    try:    
        pose_command_(args.filepath, args.roi)
    except argparse.ArgumentError as arg_err:
        print(f"Argument error: {arg_err}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: The file specified cannot be found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
