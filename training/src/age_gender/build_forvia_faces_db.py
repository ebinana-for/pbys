"""Build Forvia NIR faces database from video and smarteye csv file, for age/gender model training."""

import os
import argparse
import cv2
import pandas as pd
from src.utils.viewer_utils import get_face_bbox
import progressbar


def main(debug=False):
    """Main function to extract faces from Forvia NIR video and save them with labels."""
    if debug:
        print("Debug mode enabled, will process only 2000 frames")

    consecutive_save = 0
    parser = argparse.ArgumentParser("Forvia NIR record faces extractor.")
    parser.add_argument("-v", "--video_path", required=True, help="Path to input video")
    parser.add_argument(
        "-l", "--csv_file", required=True, help="smarteye tracking csv file", type=str
    )
    parser.add_argument("-o", "--out_path", required=True, help="Path to store outputs")
    parser.add_argument("-t", "--task", required=True, help="train or test")
    args = parser.parse_args()

    assert args.task in ["train", "test"], "task argument must be train or test"
    assert os.path.isfile(args.video_path), "Video file file not found"
    cap = cv2.VideoCapture(args.video_path)
    assert cap.isOpened, "Failed to open video {}".format(args.video_path)

    # frame_width = int(cap.get(3))
    # frame_height = int(cap.get(4))
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # size = (frame_width, frame_height)

    # Load smarteye csv data
    assert os.path.isfile(args.csv_file), "smarteye csv file not found"
    df = pd.read_csv(args.csv_file)

    # Extract aged and gender from file name
    filename = os.path.basename(args.video_path)
    age_gender_data = filename.split(".")[0]
    age_gender_data = age_gender_data.split("_")
    gt_age = age_gender_data[-4]
    gt_gender = age_gender_data[-3]
    gt_gender_int = 0 if gt_gender == "M" else 1

    out_dir = os.path.join(args.out_path, "images")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Open the label csv file, or create it
    label_file = "forvia_face_{}.csv".format(args.task)

    if os.path.isfile(os.path.join(args.out_path, label_file)):
        label_df = pd.read_csv(os.path.join(args.out_path, label_file))
        label_df.drop("Unnamed: 0", axis=1, inplace=True)
        image_cnt = len(label_df)
        person_cnt = label_df["img_name"].apply(lambda x: int(x.split("_")[2])).values
        person_cnt = sorted(person_cnt)[-1].item() + 1
    else:
        label_df = None
        image_cnt = 0
        person_cnt = 0

    data_list = (
        []
    )  # img_name, age, gender, face_x0, face_y0, face_x1, face_y1, person_x0, person_y0, person_x1, person_y1

    count = 0
    max_consecutive_save = 1
    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)

    count = 0
    max_consecutive_save = 1
    video_image_cnt = 0  # NOUVEAU COMPTEUR LOCAL
    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            head_crop0, head_crop_quality0, frame2, head_bbox0 = get_face_bbox(
                df, occupant_id=0, frame=frame, frame_cnt=count
            )
            # head_crop1, head_crop_quality1, frame, _ = get_face_bbox(df,
            #                                                          occupant_id=1,
            #                                                          frame=frame,
            #                                                          frame_cnt=count)
            
            # Nouvelle extraction propre pour le passager (en récupérant bien sa bounding box au lieu de '_')
            head_crop1, head_crop_quality1, frame3, head_bbox1 = get_face_bbox(
                df, occupant_id=1, frame=frame, frame_cnt=count
            )

            face_saved_in_frame = False

            # On boucle sur le conducteur puis le passager
            for head_crop, head_bbox in [(head_crop0, head_bbox0), (head_crop1, head_bbox1)]:
                if head_crop is not None:
                    face_saved_in_frame = True
                    if consecutive_save == max_consecutive_save:
                        consecutive_save = 0
                    else:
                        img_name = "{0:}_{1:}_{2:}_{3:06d}.jpg".format(
                            gt_age, gt_gender_int, person_cnt, image_cnt
                        )
                        data_list.append(
                            [
                                img_name,
                                gt_age,
                                gt_gender,
                                head_bbox[0],
                                head_bbox[1],
                                head_bbox[0] + head_bbox[2],
                                head_bbox[1] + head_bbox[3],
                                -1,
                                -1,
                                -1,
                                -1,
                            ]
                        )
                        # Save image
                        cv2.imwrite(
                            os.path.join(out_dir, img_name),
                            frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 95],
                        )
                        consecutive_save += 1
                        image_cnt += 1
                        video_image_cnt += 1

            if not face_saved_in_frame:
                consecutive_save = 0
        else:
            break
        bar.update(count)
        count += 1

        if video_image_cnt > 1000:  # Avoid too many images from the same video
            break

        if debug and count > 2000:
            break

    cap.release()
    data_df = pd.DataFrame(
        data_list,
        columns=[
            "img_name",
            "age",
            "gender",
            "face_x0",
            "face_y0",
            "face_x1",
            "face_y1",
            "person_x0",
            "person_y0",
            "person_x1",
            "person_y1",
        ],
    )
    if label_df is not None:
        label_df = pd.concat([label_df, data_df])
        label_df.reset_index(drop=True, inplace=True)
    else:
        label_df = data_df

    # Save results
    label_df.to_csv(os.path.join(args.out_path, label_file))


if __name__ == "__main__":
    main()
