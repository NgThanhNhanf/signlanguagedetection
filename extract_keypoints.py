import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import os
from tqdm import tqdm

# ======================
# 1. Load MediaPipe
# ======================

model_path = 'hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

pose_base_options = python.BaseOptions(
    model_asset_path="pose_landmarker_full.task"
)

pose_options = vision.PoseLandmarkerOptions(
    base_options=pose_base_options,
    num_poses=1
)

pose_landmarker = vision.PoseLandmarker.create_from_options(
    pose_options
)

# ======================
# 2. Extract keypoints
# ======================

video_dir = 'archive\\Dataset\\Videos'
output_dir = 'keypoints'

os.makedirs(output_dir, exist_ok=True)

for filename in tqdm(os.listdir(video_dir)):

    # ======================
    # Check file đã extract
    # ======================

    video_name = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, video_name + '.npy')

    if os.path.exists(output_path):
        continue

    # ======================
    # Open video
    # ======================

    video_path = os.path.join(video_dir, filename)
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if fps <= 0 or n_frames <= 0:
        print(f"Không đọc được video: {filename}")
        cap.release()
        continue

    duration = min(n_frames / fps, 3.0)

    target_fps = 15
    target_frames = int(target_fps * duration)

    if target_frames <= 0:
        cap.release()
        continue

    indices = np.linspace(
        0,
        duration * fps - 1,
        target_frames
    ).astype(int)

    indices = set(indices)

    keypoints = []

    frame_idx = 0

    # ======================
    # Read frames
    # ======================

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx in indices:

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = hand_landmarker.detect(mp_image)

            frame_keypoints = np.zeros(
                (2, 21, 3),
                dtype=np.float32
            )

            for hand_id, hand_landmarks in enumerate(
                result.hand_landmarks
            ):

                handedness = result.handedness[hand_id][0]
                label = handedness.category_name

                if label == "Left":
                    slot = 0

                elif label == "Right":
                    slot = 1

                else:
                    continue

                for landmark_id, landmark in enumerate(
                    hand_landmarks
                ):

                    frame_keypoints[
                        slot,
                        landmark_id
                    ] = [
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ]

            keypoints.append(frame_keypoints)

        frame_idx += 1

    cap.release()

    # ======================
    # Save
    # ======================

    np.save(
        output_path,
        np.array(keypoints, dtype=np.float32)
    )