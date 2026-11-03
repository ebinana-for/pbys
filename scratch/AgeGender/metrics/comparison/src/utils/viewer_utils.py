"""Utility functions for processing and visualizing face and body keypoints in a video frame."""

import numpy as np
import math
import cv2


def compute_roll_and_yaw(landmarks):
    """
    Calculate the roll angle using the angle between the eyes
    landmarks = [[xl,yl]    # Left eye coordinate
                 [xr, yr]   # Right coordinate
                 [xn, yn]]  # nose coordinate

    """
    l_eye, r_eye, nose = landmarks
    if l_eye is not None and r_eye is not None:
        roll_angle = np.degrees(
            np.arctan2(
                landmarks[1][1] - landmarks[0][1], landmarks[1][0] - landmarks[0][0]
            )
        )
    else:
        roll_angle = None  # by defaut, if eyes are not detected
    yaw_l = None
    yaw_r = None
    if nose is None:
        pass
    else:
        # Calculate the yaw angle using the angle between the eyes and the tip of the nose
        yaw_l = (
            np.degrees(
                np.arctan2(
                    landmarks[0][1] - landmarks[2][1], landmarks[0][0] - landmarks[2][0]
                )
            )
            if l_eye is not None
            else None
        )
        yaw_r = (
            np.degrees(
                np.arctan2(
                    landmarks[1][1] - landmarks[2][1], landmarks[1][0] - landmarks[2][0]
                )
            )
            if r_eye is not None
            else None
        )

    return roll_angle, yaw_l, yaw_r


def compute_face_bbox(landmarks):
    """
    Calculate the face bounding box using facial landmarks.
    landmarks = [[xl,yl]    # Left eye
                 [xr, yr]   # Right eye
                 [xn, yn]   # Chin coordinate
                 [xle, yle] # Left ear
                 [xre, yre] # Right ear
    """
    # Nose keypoint location is not reliable (tracking delay), so use only eyes
    l_eye, r_eye, chin, l_ear, r_ear = landmarks

    minInterEyeDistance = 5.0
    minWidthFaceCrop = 250
    minHeightFaceCrop = 350

    bbox = None

    # Check if all required keypoints are present
    if (
        l_eye is not None
        and r_eye is not None
        and l_ear is not None
        and r_ear is not None
        and chin is not None
    ):  # Best case

        # Calculate distance between eyes
        eye_dist = math.sqrt((l_eye[0] - r_eye[0]) ** 2 + (l_eye[1] - r_eye[1]) ** 2)

        # Filter out if eyes are too close
        if eye_dist < minInterEyeDistance:
            # print(f'Filtered eyes too close')
            return bbox

        # Check if chin is between eyes horizontally
        if (chin[0] < l_eye[0]) or (chin[0] > r_eye[0]):
            # print('Filtered chin not between eyes')
            return bbox

        # Check if eyes have shifted compared to ears (right shift)
        if (l_ear[0] > r_eye[0]) and (r_ear[0] > r_eye[0]):
            # print('Filtered eyes shifted right compared to ears')
            return bbox

        # Check if eyes have shifted compared to ears (left shift)
        if (l_ear[0] < l_eye[0]) and (r_ear[0] < l_eye[0]):
            # print('Filtered eyes shifted left compared to ears')
            return bbox

        # Calculate vertical distance from eyes to chin
        eye_to_chin = chin[1] - (r_eye[1] + l_eye[1]) / 2.0

        # Filter out if chin is too close to eyes
        if eye_to_chin <= (eye_dist * 1.5):
            # print(f'Filtered chin too close to eyes')
            return bbox

        # Calculate center of face using ears for x, eyes for y
        center_x = (l_ear[0] + r_ear[0]) / 2.0
        center_y = (l_eye[1] + r_eye[1]) / 2.0

        # Estimate width and height of the face crop
        width = eye_dist * 3.2
        height = eye_to_chin * 2.2

        # Calculate bounding box coordinates
        facecrop_xmin = int(center_x - width / 2.0 + 0.5)
        facecrop_ymin = int(center_y - 0.8 * height / 2.0 + 0.5)
        facecrop_xmax = int(center_x + width / 2.0 + 0.5)
        facecrop_ymax = int(center_y + height / 2.0 + 0.5)

        width = facecrop_xmax - facecrop_xmin
        height = facecrop_ymax - facecrop_ymin

        ratio = width / height

        # Filter out bounding boxes with bad aspect ratio or too small
        if (
            ratio < 0.6
            or ratio > 2.0
            or width < minWidthFaceCrop
            or height < minHeightFaceCrop
        ):
            # print(f'Filtered facebbox for bad crop size')
            return bbox

        quality = 1  # Set quality flag

        bbox = [facecrop_xmin, facecrop_xmax, facecrop_ymin, facecrop_ymax, quality]

    else:
        # print("Filtered face bbox: missing keypoint")
        pass

    return bbox


def get_face_bbox(df_data, occupant_id=0, frame=None, frame_cnt=0):
    """
    Extracts the face bounding box and cropped face image from the frame using body keypoints.

    Args:
        df_data: DataFrame containing body keypoints data.
        occupant_id: ID of the occupant to process.
        frame: The image frame to crop from.
        frame_cnt: The frame index to use for keypoint extraction.

    Returns:
        head_crop: Cropped face image (or None if not found).
        head_crop_quality: Quality flag for the crop (0 or 1).
        None: Placeholder for future use.
        head_bbox: Bounding box coordinates and quality (or None if not found).
    """

    minBodyPtQuality = 0.1  # Minimum quality threshold for keypoints

    # Get required body keypoints: left eye, right eye, chin, left ear, right ear
    face_landmarks = []
    row = frame_cnt

    # Extract x, y, and quality for all 28 keypoints
    body_kpt_x = [
        int(
            df_data[
                "/trackingState/occupants/{}/bodyKeyPoints/{}/value/0".format(
                    occupant_id, ii
                )
            ].loc[row]
        )
        for ii in range(28)
    ]
    body_kpt_y = [
        int(
            df_data[
                "/trackingState/occupants/{}/bodyKeyPoints/{}/value/1".format(
                    occupant_id, ii
                )
            ].loc[row]
        )
        for ii in range(28)
    ]
    body_kpt_quality = [
        df_data[
            "/trackingState/occupants/{}/bodyKeyPoints/{}/quality".format(
                occupant_id, ii
            )
        ].loc[row]
        for ii in range(28)
    ]

    # Left eye
    if body_kpt_quality[1] > minBodyPtQuality:  # 1 = left eye keypoint
        face_landmarks.append([body_kpt_x[1], body_kpt_y[1]])
    else:
        face_landmarks.append(None)

    # Right eye
    if body_kpt_quality[2] > minBodyPtQuality:  # 2 = right eye keypoint
        face_landmarks.append([body_kpt_x[2], body_kpt_y[2]])
    else:
        face_landmarks.append(None)

    # Chin
    if body_kpt_quality[25] > minBodyPtQuality:  # 25 = chin keypoint
        face_landmarks.append([body_kpt_x[25], body_kpt_y[25]])
    else:
        face_landmarks.append(None)

    # Left ear
    if body_kpt_quality[3] > minBodyPtQuality:  # 3 = left ear keypoint
        face_landmarks.append([body_kpt_x[3], body_kpt_y[3]])
    else:
        face_landmarks.append(None)

    # Right ear
    if body_kpt_quality[4] > minBodyPtQuality:  # 4 = right ear keypoint
        face_landmarks.append([body_kpt_x[4], body_kpt_y[4]])
    else:
        face_landmarks.append(None)

    # print(f"Face landmarks: {face_landmarks}")

    # Compute the face bounding box using the extracted landmarks
    head_bbox = compute_face_bbox(face_landmarks)

    head_crop_quality = 0
    head_crop = None

    # If a valid bounding box is found and all coordinates are positive
    if head_bbox is not None and (np.array(head_bbox) > 0).all():
        try:
            # Crop the face region from the frame
            head_crop = frame[
                head_bbox[2] : head_bbox[3],
                head_bbox[0] : head_bbox[1],
            ]
            head_crop_quality = head_bbox[4]

            # If the crop is empty, set to None
            if not len(head_crop):
                print("No head crop found, returning None.")
                head_crop = None
                head_crop_quality = 0

        except IndexError:
            # If the bbox is out of the frame, head_crop will be None
            head_crop = None
            head_crop_quality = 0
    else:
        pass
        # print("No head bbox found, cannot crop the face.")

    return head_crop, head_crop_quality, None, head_bbox


def draw_body_keypoints2D(df_data, frame, occupant_id=0, frame_cnt=0):

    raw = frame_cnt
    body_kpt_x = [
        int(
            df_data[
                "/trackingState/occupants/{}/bodyKeyPoints/{}/value/0".format(
                    occupant_id, ii
                )
            ].loc[raw]
        )
        for ii in range(28)
    ]
    body_kpt_y = [
        int(
            df_data[
                "/trackingState/occupants/{}/bodyKeyPoints/{}/value/1".format(
                    occupant_id, ii
                )
            ].loc[raw]
        )
        for ii in range(28)
    ]
    body_kpt_quality = [
        int(
            df_data[
                "/trackingState/occupants/{}/bodyKeyPoints/{}/quality".format(
                    occupant_id, ii
                )
            ].loc[raw]
        )
        for ii in range(28)
    ]

    for idx, (xx, yy, quality) in enumerate(
        zip(body_kpt_x, body_kpt_y, body_kpt_quality)
    ):
        color = (0, 255, 0) if quality > 0 else (0, 0, 255)
        frame = cv2.circle(frame, (xx, yy), 6, color, 4)
        frame = cv2.putText(
            frame,
            "{}".format(idx),
            (xx, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            1,
            cv2.LINE_AA,
        )

    return frame
