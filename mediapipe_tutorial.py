import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np

model_path = 'hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options = BaseOptions(model_asset_path = model_path),
    running_mode = VisionRunningMode.IMAGE,
    num_hands = 2
)

landmarker = vision.HandLandmarker.create_from_options(options)

img = cv2.imread('baa2aac9b3a033fe6ab1.jpg')

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=img_rgb
)


result = landmarker.detect(mp_image)

print(f'Số bàn tay: {len(result.hand_landmarks)}')

for hand_id, hand_landmarks in enumerate(result.hand_landmarks):

    print(f"\n--- Hand {hand_id} ---")

    for landmark_id, landmark in enumerate(hand_landmarks):

        print(
            f"Landmark {landmark_id}: "
            f"x={landmark.x:.4f}, "
            f"y={landmark.y:.4f}, "
            f"z={landmark.z:.4f}"
        )

keypoints = []

for hand_landmark in result.hand_landmarks:
    for landmark in hand_landmark:
        keypoints.append([landmark.x, landmark.y, landmark.z])

keypoints = np.array(keypoints)

print(keypoints.shape)
print(keypoints)