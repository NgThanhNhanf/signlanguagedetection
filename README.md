Dataset:
https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed

Pipeline:
Frames -> MediaPipe (extract keypoint) -> LSTM / Transformer / SignBART -> Text / Audio

Paper SignBART:
D:\Projects\SignLanguageDetection\2506.21592v1.pdf


Train Pipe:
Video (frames) -> MediaPipe -> Keypoints -> Numpy Array -> Train# signlanguagedetection
