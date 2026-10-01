import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================
# 1. Load MediaPipe model
# =========================

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

# =========================
# 2. Open camera
# =========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở camera")
    exit()


frame_timestamp_ms = 0


# =========================
# 3. Process video
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Không đọc được frame")
        break

    # OpenCV: BGR
    # MediaPipe: RGB

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Convert OpenCV image → MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Timestamp phải tăng dần
    frame_timestamp_ms += 33

    # Detect
    result = detector.detect_for_video(
        mp_image,
        frame_timestamp_ms
    )


    # =========================
    # 4. Draw landmarks
    # =========================

    if result.hand_landmarks:

        for hand_landmarks in result.hand_landmarks:

            # Draw points
            for landmark in hand_landmarks:

                h, w, _ = frame.shape

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1
                )

            # Draw connections
            connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

            for connection in connections:

                start = connection.start
                end = connection.end

                h, w, _ = frame.shape

                x1 = int(
                    hand_landmarks[start].x * w
                )
                y1 = int(
                    hand_landmarks[start].y * h
                )

                x2 = int(
                    hand_landmarks[end].x * w
                )
                y2 = int(
                    hand_landmarks[end].y * h
                )

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )
    

    # =========================
    # 5. Display
    # =========================

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