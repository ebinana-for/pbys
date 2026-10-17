"""
This script is used for displaying online posture classification results
(+gt if provided) on a video
"""

import os
import argparse
import cv2
import numpy as np
import pandas as pd

import pdb

GT_LABEL_MAP = {
    0: "Rounded_back",
    1: "Too_close_to_airbag",
    2: "Pelvis rotation",
    3: "Torso_dissimetry",
    4: "One_shoulder_in_contact",
    5: "Pelvis_drift",
    6: "Legs_on_dashboard",
    7: "Reaching_back",
    8: "Looking_left_right",
    9: "Looking_back",
    10: "Lateral_body_contact",
    11: "Looking_straight",
    12: "Correctly_seated",
    13: "Buttock forward",
}

PRED_LABEL_MAP = {
    0: "Correctly seated",
    1: "Too close to airbag",
    2: "Legs on dashboard",
    3: "Torso dissimetry",
    4: "Reaching back",
    5: "Lateral body contact",
    6: "One shoulder in contact",
    7: "Pelvis rotation",
    8: "Pelvis drift",
}

GT2PRED_MAP = {
    0: 0,
    1: 1,
    2: 7,
    3: 3,
    4: 6,
    5: 8,
    6: 2,
    7: 4,
    8: 0,
    9: 0,
    10: 5,
    11: 0,
    12: 0,
    13: 0,
}

# COLORMAP = {0:(0.255,128),
#             1:(0,128,255),
#             2:(0.255,255),
#             3:(0,0,255),
#             4:(255,255,0),
#             5:(255,128,0),
#             6:(255,0,0),
#             7:(255,0,128),
#             8:(255,0,255)}


def main(video_file, pred_file, gt_file, out_dir=None):
    """
    Main function to run the posture viewer.
    Args:
        video_file (str): Path to the input video file.
        pred_file (str): Path to the input prediction file (CSV).
        gt_file (str): Path to the input ground truth file (optional).
        out_dir (str): Output directory to save the resulting video (optional).
    Returns:
        None
    This function performs the following steps:
    1. Loads the predictions from the CSV file and processes them.
    2. Loads the ground truth labels from the specified file (if provided).
    3. Calls the `run` function to process the video and display predictions and ground truth labels.
    """
    # sanity checks
    assert os.path.isfile(video_file), "Video file not found: {}".format(video_file)
    assert os.path.isfile(pred_file), "Prediction file not found: {}".format(pred_file)

    # Load predictions
    thresh = 0.5
    raw_pred = pd.read_csv(pred_file) if pred_file is not None else None
    raw_pred.drop(columns=["index"], inplace=True)
    pred_array = np.zeros((len(raw_pred), len(PRED_LABEL_MAP)))
    pred_array[:, 1:] = (raw_pred.values > thresh).astype(np.int64)
    pred_array[np.sum(pred_array[:, 1:], axis=1) == 0, 0] = 1
    pred_array[np.sum(pred_array[:, :], axis=1) < 0, :] = 0
    pred_array[pred_array < 0] = 0

    # Load gt
    gt_array = np.zeros((len(raw_pred), len(PRED_LABEL_MAP)))
    label_df_list = []
    if gt_file is not None:
        raw_gt_df = pd.read_csv(
            gt_file, names=["file", "0", "1", "2", "3", "4"], delimiter=r"\s+"
        )
        raw_gt_df["frame_index"] = raw_gt_df["file"].apply(
            lambda x: int(os.path.splitext(os.path.basename(x))[0][-6:])
        )

        for ii in GT_LABEL_MAP.keys():
            df_label_tmp = (raw_gt_df[["0", "1", "2", "3", "4"]] == ii).any(axis=1)
            gt_array[df_label_tmp.values, GT2PRED_MAP[ii]] = 1
            label_df_list.append(df_label_tmp)

    run(video_file, pred_array, gt_array, out_dir=out_dir)


def run(video_file, pred_array, gt_array, out_dir=None):
    """
    Run the posture viewer on the input video file.
    Args:
        video_file (str): Path to the input video file.
        pred_array (np.ndarray): Array of predictions for each frame.
        gt_array (np.ndarray): Array of ground truth labels for each frame (optional).
        out_dir (str): Output directory to save the resulting video (optional).
    Returns:
        None
    """
    # Open input video
    video_in = cv2.VideoCapture(video_file)
    if not video_in.isOpened():
        print("Cannot open video")
        return
    fps = video_in.get(cv2.CAP_PROP_FPS)
    width = video_in.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video_in.get(cv2.CAP_PROP_FRAME_HEIGHT)
    size = (int(width / 2), int(height / 2))

    # Create output video
    in_file = os.path.basename(video_file)

    out_file = in_file.split(".")[0] + "_posture_viewer_tidl.mp4"
    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, out_file)
    else:
        out_path = out_file
    video_out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    frame_idx = 0
    # dt = 1.0/fps
    # segment_speed = 500 # px/sec
    while True:
        ret, frame_in = video_in.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Write label and pred in the image
        frame_out = frame_in.copy()
        frame_out = cv2.resize(frame_out, size, interpolation=cv2.INTER_AREA)

        # pdb.set_trace()
        pred_class = pred_array[frame_idx, :]
        pred_labels = [
            PRED_LABEL_MAP[ii] for ii, val in enumerate(pred_class) if val == 1
        ]
        posx = size[0] // 2
        posy = 50
        frame_out = cv2.putText(
            frame_out,
            "Prediction",
            (posx, posy),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        for pred_label in pred_labels:
            posy += 50
            frame_out = cv2.putText(
                frame_out,
                pred_label,
                (posx, posy),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        if gt_array is not None:
            labels = gt_array[frame_idx, :]
            # If at least one position is annotated, set "correctly seated" to 0
            if np.sum(labels[1:]) > 0:
                labels[0] = 0
            gt_labels0 = [ii for ii, val in enumerate(labels) if val == 1]
            gt_labels = [PRED_LABEL_MAP[ii] for ii in gt_labels0]

            posx = 50
            posy = 50
            frame_out = cv2.putText(
                frame_out,
                "Ground Truth",
                (posx, posy),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            for gt_label in gt_labels:
                posy += 50
                frame_out = cv2.putText(
                    frame_out,
                    gt_label,
                    (posx, posy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.4,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            frame_out = cv2.putText(
                frame_out,
                str(frame_idx),
                (size[0] - 100, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        video_out.write(frame_out)
        frame_idx += 1

    video_in.release()
    video_out.release()


# def plotTimeSegments(frame_out, array, frame_idx, dt, segment_speed, type):

#     # Select the useful part of array
#     pdb.set_trace()
#     array_hor = int((frame_out.shape[1]//(segment_speed*dt))//2)
#     array_start_idx = frame_idx - array_hor
#     array_stop_idx = frame_idx + array_hor

#     indices_ok = np.array([ii for ii in range(array_start_idx, array_stop_idx)])
#     array_usefull = array[indices_ok,:]
#     array_usefull[indices_ok<0,:] = 0

#     nb_seg_to_plot = int(np.max(np.sum(array_usefull, axis=1)))
#     if type == 'pred':
#         offset = 40
#     else:
#         offset = 150

#     for ii in range(array_usefull.shape[1]):
#         idx_plot = np.where(array_usefull[:,ii]>0)[0].tolist()

#         pdb.set_trace()
#         if len(idx_plot)>0:
#             # Split in contiguous sequences
#             idx_plot_seq = []
#             for _, g in groupby(enumerate(idx_plot), lambda x: x[0] - x[1]):
#                 idx_plot_seq.append([v for _, v in g])

#             # Draw segments
#             for seg in idx_plot_seq:
#                 col_start = seg[0] * (frame_out.shape[1]//2) / array_hor
#                 col_end = seg[-1] * (frame_out.shape[1]//2) / array_hor

#                 col_start = np.clip(col_start, 0, frame_out.shape[1])
#                 col_end = np.clip(col_end, 0, frame_out.shape[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Posture video viewer")
    parser.add_argument(
        "-v", "--video_file", required=False, help="Path to the input video"
    )
    parser.add_argument(
        "-p",
        "--pred_file",
        required=True,
        help="Path to input csv posture prediction file",
    )
    parser.add_argument(
        "-g",
        "--gt_file",
        required=False,
        default=None,
        help="Path to the input ground truth file (default.txt)",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        required=False,
        default=None,
        help="Output directory (optionnal)",
    )
    args = parser.parse_args()
    main(args.video_file, args.pred_file, args.gt_file)

    # video_file = "/media/jdefretin/DATA/Forvia/data/berkan_abaci_29_M_186_78.mp4"
    # pred_file = "/media/jdefretin/DATA/Forvia/forvia-oms-library/data/videos/position_2025-05-10T13h16m06s_epoch_43/berkan_abaci_29_M_186_78_out_positions.csv"
    # gt_file = "/media/jdefretin/DATA/PSA_domain_adaptation/sshfs_mountpoint_kirby_mnt_data/forvia/private_datasets/recordings_forvia_30_01/test/labels/berkan_abaci_29_M_186_78/default.txt"
    video_file = args.video_file
    pred_file = args.pred_file
    gt_file = args.gt_file
    out_dir = args.out_dir
    main(video_file, pred_file, gt_file, out_dir=out_dir)
