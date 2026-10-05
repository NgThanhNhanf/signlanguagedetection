import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import numpy as np
import cv2
import os
from tqdm import tqdm


# ======================
# 1. Load MediaPipe Pose
# ======================

model_path = "pose_landmarker_full.task"

pose_base_options = python.BaseOptions(
    model_asset_path=model_path
)

pose_options = vision.PoseLandmarkerOptions(
    base_options=pose_base_options,
    num_poses=1
)

pose_landmarker = vision.PoseLandmarker.create_from_options(
    pose_options
)


# ======================
# 2. Paths
# ======================

video_dir = "wlasl/archive/videos"
output_dir = "pose_keypoints"

os.makedirs(output_dir, exist_ok=True)


# ======================
# 3. Pose landmark IDs
# ======================

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14

POSE_IDS = [
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW
]


# ======================
# 4. Extract
# ======================

for filename in tqdm(os.listdir(video_dir)):

    # ======================
    # Check đã extract
    # ======================

    video_name = os.path.splitext(filename)[0]

    output_path = os.path.join(
        output_dir,
        video_name + ".npy"
    )

    if os.path.exists(output_path):
        continue


    # ======================
    # Open video
    # ======================

    video_path = os.path.join(
        video_dir,
        filename
    )

    cap = cv2.VideoCapture(video_path)

    # ======================
    # Read frames
    # ======================

    keypoints = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # ======================
        # BGR → RGB
        # ======================

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ======================
        # MediaPipe Image
        # ======================

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )


        # ======================
        # Pose detection
        # ======================

        result = pose_landmarker.detect(mp_image)


        # ======================
        # [4, 3]
        # ======================

        frame_keypoints = np.zeros(
            (4, 3),
            dtype=np.float32
        )


        # ======================
        # Get pose landmarks
        # ======================

        if len(result.pose_landmarks) > 0:

            landmarks = result.pose_landmarks[0]

            for i, landmark_id in enumerate(POSE_IDS):

                landmark = landmarks[landmark_id]

                frame_keypoints[i] = [
                    landmark.x,
                    landmark.y,
                    landmark.z
                ]


        keypoints.append(frame_keypoints)


    cap.release()


    # ======================
    # Save
    # ======================

    keypoints = np.array(
        keypoints,
        dtype=np.float32
    )

    np.save(
        output_path,
        keypoints
    )