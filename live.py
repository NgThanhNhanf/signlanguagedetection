import cv2
import torch
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from baseline_model import BaselineModel

import numpy as np


model = BaselineModel(126, 256, 2, 3315)
model.load_state_dict(
    torch.load(
        'result/lstm/best_model.pth',
        map_location='cpu'
    )
)

MODEL_PATH = "hand_landmarker.task"

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở camera")
    exit()


frame_timestamp_ms = 0

keypoints = []

while True:

    ret, frame = cap.read()

    if not ret:
        print("Không đọc được frame")
        break

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    frame_timestamp_ms += 33

    result = detector.detect_for_video(
        mp_image,
        frame_timestamp_ms
    )

    frame_keypoints = np.zeros((2, 21, 3))
    if result.hand_landmarks:

        for hand_id, hand_landmarks in enumerate(result.hand_landmarks):

            for landmark_id, landmark in enumerate(hand_landmarks):

                h, w, _ = frame.shape

                x = int(landmark.x * w)
                y = int(landmark.y * h)
                z = landmark.z

                frame_keypoints[hand_id, landmark_id, 0] = x
                frame_keypoints[hand_id, landmark_id, 1] = y        
                frame_keypoints[hand_id, landmark_id, 2] = z       

        keypoints.append(frame_keypoints)
    else:
        keypoints.clear()

    if len(keypoints) == 45:
        key_tensor = torch.tensor(keypoints, dtype = torch.float32)
        key_tensor = key_tensor.unsqueeze(0)
        B, T, _, _, _ = key_tensor.shape
        key_tensor = key_tensor.reshape(1, T, -1)
        #-------------- Call Model ---------------#
        outputs = model(key_tensor)
        print(outputs.argmax(dim = 1))

    cv2.imshow(
        "MediaPipe Hand Tracking",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================
# 6. Cleanup
# =========================

cap.release()
cv2.destroyAllWindows()

detector.close()