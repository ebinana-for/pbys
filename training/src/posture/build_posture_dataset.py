""" Build a dataset for posture classification from MPEG videos and labels."""
import os
import argparse
import cv2
import numpy as np
import pandas as pd
import progressbar
import glob


def main():
    """Main function to build the posture classification dataset."""
    # Parse command line arguments
    # Usage: python build_posture_dataset.py -m <mpeg_root_dir> -l <label_root_dir> -o <out_path>
    # Example: python build_posture_dataset.py -m /path/to/mpeg -l /path/to/labels -o /path/to/output

    parser = argparse.ArgumentParser("Build position classification dataset")
    parser.add_argument(
        "-m", "--mpeg_root_dir", required=False, help="Directory containing mpeg videos"
    )
    parser.add_argument(
        "-l", "--label_root_dir", required=False, help="Directory containing labels"
    )
    parser.add_argument(
        "-o", "--out_path", required=False, help="Path to store output video"
    )
    args = parser.parse_args()

    # Scan directories and get paths
    mpeg_list = glob.glob(args.mpeg_root_dir + "/*.mp4")
    label_list0 = glob.glob(args.label_root_dir + "/*")

    # Sanity check
    label_list = []
    for label_dir in label_list0:
        file1_exist = os.path.isfile(label_dir + "/default.txt")
        file2_exist = os.path.isfile(label_dir + "/synsets.txt")
        if file1_exist and file2_exist:
            label_list.append(label_dir)

    label_list_final = label_list
    label_names_ref = None
    for video_idx, label_dir in enumerate(label_list_final):

        # Read label file and get the list of frames to be extracted
        label_data = pd.read_csv(
            label_dir + "/default.txt",
            names=["file", "0", "1", "2", "3", "4"],
            delimiter=r"\s+",
        )
        label_data_keep = label_data[
            ~label_data[["0", "1", "2", "3", "4"]].isnull().all(axis=1)
        ].copy()
        label_data_keep["frame_idx"] = label_data_keep["file"].apply(
            lambda x: int(x[6:12])
        )

        label_names = pd.read_csv(label_dir + "/synsets.txt", names=["name"])
        if label_names_ref is None:
            label_names_ref = np.squeeze(label_names.values).tolist()
        else:
            label_names = np.squeeze(label_names.values).tolist()
            assert (
                label_names_ref == label_names
            ), "Label definition issue for {}".format(label_dir)

        # Get the corresponding video
        basename = os.path.basename(label_dir)
        mpeg_file = [elt for elt in mpeg_list if basename in elt]

        if len(mpeg_file) == 0:
            print("Video for {} not found".format(basename))
            continue
        if len(mpeg_file) > 1:
            print("{} video for {} found".format(len(mpeg_file), basename))
            continue

        out_dir = os.path.join(args.out_path, basename)

        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        cap = cv2.VideoCapture(mpeg_file[0])
        frames_indices = label_data_keep["frame_idx"].values
        count = 0
        print(
            "Processing video {} of {}: {}".format(
                video_idx, len(label_list_final), basename
            )
        )
        bar = progressbar.ProgressBar(
            maxval=frames_indices.max().item() + 1
        )  # Initialize the progress bar
        while cap.isOpened():
            ret, frame = cap.read()
            if count in frames_indices:
                assert ret, "error reading frame {} for video {}".format(
                    count, basename
                )
                output_file = "frame_{0:06d}.jpg".format(count)
                out_path = os.path.join(out_dir, output_file)
                if not os.path.isfile(out_path):
                    cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            count += 1
            if count > frames_indices.max().item():
                break
            bar.update(count)
        bar.finish()
        cap.release()


if __name__ == "__main__":
    main()
