"""Age/gender model inference on a video file, with visualization and prediction logging"""

import os
import argparse
import cv2
import numpy as np
import pandas as pd
import json
from src.utils.viewer_utils import (
    get_face_bbox,
    draw_body_keypoints2D,
    draw_body_keypoints3D,
)
import progressbar


def process_gender_time_pred(gender_list):
    if len(gender_list) == 0:
        return (None, None)
    gender_list = np.array(gender_list)
    gender_list = gender_list[
        gender_list[:, -1] > 0, :
    ]  # Revome entries if head crop quality=0
    gender_average = np.sum(gender_list[:, 0] * gender_list[:, 1]) / np.sum(
        gender_list[:, 1]
    )
    confidence = float(np.maximum(gender_average, 1 - gender_average))
    return ("H", confidence) if gender_average < 0.5 else ("F", confidence)


def process_age_time_pred(age_list):
    if len(age_list) == 0:
        return (None, None)
    age_list = np.array(age_list)
    age_list = age_list[age_list[:, -1] > 0, :]  # Revome entries if head crop quality=0
    age_list = age_list[
        age_list[:, 1] < 12.0, :
    ]  # Revome entries if prediction std is too high

    std_sig_inv = 1 - (1.0 / (1.0 + np.exp(-age_list[:, 1] / 10.0)))
    std_sig_inv /= np.sum(std_sig_inv)
    age_average = float(np.sum(std_sig_inv * age_list[:, 0]))
    age_var = np.sum((std_sig_inv * (age_list[:, 0] - age_average) ** 2))
    age_std = float(np.sqrt(age_var))
    return (age_average, age_std)


def load_camera_calib(calib_file):
    return


def main():

    parser = argparse.ArgumentParser("Age and gender video inference")
    parser.add_argument(
        "-m", "--model_path", required=False, help="Path to input model"
    )
    parser.add_argument(
        "-v", "--video_path", required=False, help="Path to input video"
    )
    parser.add_argument(
        "-c", "--camera_id", required=False, help="camera id, local only", type=int
    )
    parser.add_argument(
        "-l", "--csv_file", required=False, help="smarteye tracking csv file", type=str
    )
    parser.add_argument(
        "-s", "--show", action="store_true", help="show stream, local only"
    )
    parser.add_argument(
        "-o", "--out_path", required=False, help="Path to store output video"
    )
    args = parser.parse_args()
    # detector = insightface.model_zoo.get_model('../ressources/det_500m.onnx')
    # detector.prepare(ctx_id=0, input_size=(640, 640))

    if args.model_path:
        from src.age_gender.inference import InferenceEngineAgeGender

        age_gender_inference = InferenceEngineAgeGender(args.model_path)

    if args.video_path:
        cap = cv2.VideoCapture(args.video_path)
    elif args.camera_id is not None:
        cap = cv2.VideoCapture(args.camera_id)
    else:
        print("please set video_path or camera_id")
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Load smarteye csv data
    assert os.path.isfile(args.csv_file), "smarteye csv file not found"
    df = pd.read_csv(args.csv_file)

    # Load Camera calibration
    calib_file = os.path.dirname(os.path.dirname(args.csv_file)) + "/system.json"
    with open(calib_file) as f:
        calib = json.load(f)
    intrinsics = np.array(
        [
            [
                calib["cameras"][0]["lens"]["focalLengthX"],
                0,
                calib["cameras"][0]["lens"]["principalPointX"],
            ],
            [
                0,
                calib["cameras"][0]["lens"]["focalLengthY"],
                calib["cameras"][0]["lens"]["principalPointY"],
            ],
            [0, 0, 1],
        ]
    )
    # extrinsics = np.array(calib["cameras"][0]["transform"])

    # Compute the side of each occupant
    x_0 = float(
        np.mean(df["/trackingState/occupants/0/bodyKeyPoints/0/value/0"].values)
    )
    x_1 = float(
        np.mean(df["/trackingState/occupants/1/bodyKeyPoints/0/value/0"].values)
    )

    pos0 = frame_width // 3 if x_0 < frame_width / 2 else 2 * frame_width // 3
    pos1 = frame_width // 3 if x_1 < frame_width / 2 else 2 * frame_width // 3

    if x_0 == 0:
        pos0 = None
    if x_1 == 0:
        pos1 = None

    # print([c  for c in df.columns if "bodyKeyPoints" in c])

    # ldmarks_headers = [head for head in df.columns if "0/bodyKeyPoints/" in head]
    if args.out_path:
        out_dir = args.out_path
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (frame_width//4, frame_height//4)

        out_file = (
            os.path.splitext(os.path.basename(args.video_path))[0] + "_processed.mp4"
        )
        result = cv2.VideoWriter(
            os.path.join(out_dir, out_file), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
        )

    count = 0
    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
    age_list0 = []
    gender_list0 = []
    age_list1 = []
    gender_list1 = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            head_crop0, head_crop_quality0, frame, _ = get_face_bbox(
                df, occupant_id=0, frame=frame, frame_cnt=count
            )
            head_crop1, head_crop_quality1, frame, _ = get_face_bbox(
                df, occupant_id=1, frame=frame, frame_cnt=count
            )

            # Draw body keypoints (2D)
            frame = draw_body_keypoints2D(df, frame, occupant_id=0, frame_cnt=count)
            frame = draw_body_keypoints2D(df, frame, occupant_id=1, frame_cnt=count)

            # Draw body keypoints (3D)
            frame = draw_body_keypoints3D(
                df, intrinsics, frame, occupant_id=0, frame_cnt=count
            )
            frame = draw_body_keypoints3D(
                df, intrinsics, frame, occupant_id=1, frame_cnt=count
            )

            # Model inference
            if args.model_path:
                pred0 = (
                    age_gender_inference([head_crop0])[0]
                    if head_crop0 is not None
                    else None
                )
                pred1 = (
                    age_gender_inference([head_crop1])[0]
                    if head_crop1 is not None
                    else None
                )

                if pred0 is not None:
                    gender_list0.append(
                        [{"M": 0, "F": 1}[pred0[0]], pred0[1], head_crop_quality0]
                    )
                    age_list0.append([pred0[2], pred0[3], head_crop_quality0])
                if pred1 is not None:
                    gender_list1.append(
                        [{"M": 0, "F": 1}[pred1[0]], pred1[1], head_crop_quality1]
                    )
                    age_list1.append([pred1[2], pred1[3], head_crop_quality1])
            else:
                pred0 = pred1 = None
            if args.out_path:
                # Add prediction to frame

                if pos0 is not None:
                    posx, posy = pos0, frame.shape[0] // 4
                    frame = cv2.putText(
                        frame,
                        "occupant 0",
                        (posx, posy - 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2.0,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                    if pred0 is not None:
                        frame = cv2.putText(
                            frame,
                            pred0[0],
                            (posx, posy),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2.0,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )
                        frame = cv2.putText(
                            frame,
                            "conf. {0:.2f}".format(pred0[1]),
                            (posx + 100, posy),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2.0,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )
                        frame = cv2.putText(
                            frame,
                            "{0:1d}".format(int(pred0[2] + 0.5)),
                            (posx, posy + 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2.0,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )
                        frame = cv2.putText(
                            frame,
                            "std. {0:.1f}".format(pred0[3]),
                            (posx + 100, posy + 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2.0,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )
                if pos1 is not None:
                    posx, posy = pos1, frame.shape[0] // 4
                    frame = cv2.putText(
                        frame,
                        "occupant 1",
                        (posx, posy - 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2.0,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                    if pred1 is not None:
                        frame = cv2.putText(
                            frame,
                            pred1[0],
                            (posx, posy),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2.0,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )
                        frame = cv2.putText(
                            frame,
                            "conf. {0:.2f}".format(pred1[1]),
                            (posx + 100, posy),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2.0,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )
                        frame = cv2.putText(
                            frame,
                            "{0:1d}".format(int(pred1[2] + 0.5)),
                            (posx, posy + 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2.0,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )
                        frame = cv2.putText(
                            frame,
                            "std. {0:.1f}".format(pred1[3]),
                            (posx + 100, posy + 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2.0,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                result.write(frame)
            # if args.show:
            #     cv2.imshow("Image", cv2.resize(frame,(1720,1080)))
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break
        else:
            break
        bar.update(count)
        count += 1

    cap.release()
    if args.out_path:
        result.release()
    if args.show:
        cv2.destroyAllWindows()

    # Process age and gender estimation over time
    gender0, gender_conf0 = process_gender_time_pred(gender_list0)
    gender1, gender_conf1 = process_gender_time_pred(gender_list1)

    age0, age_std0 = process_age_time_pred(age_list0)
    age1, age_std1 = process_age_time_pred(age_list1)

    # save results
    if args.out_path:
        df_pred = pd.DataFrame(
            data={
                "gender0": [gender0],
                "age0": [age0],
                "gender_conf0": [gender_conf0],
                "age_std0": [age_std0],
                "gender1": [gender1],
                "age1": [age1],
                "gender_conf1": [gender_conf1],
                "age_std1": [age_std1],
            }
        )

        df_pred_file = (
            os.path.splitext(os.path.basename(args.video_path))[0] + "_prediction.csv"
        )
        df_pred.to_csv(os.path.join(out_dir, df_pred_file))


if __name__ == "__main__":
    main()
