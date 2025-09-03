import numpy as np
import cv2
import onnxruntime as ort

LANDMARK_IDX_68_TO_5 = [38, 43, 30, 48, 54]


# Taken from https://github.com/cunjian/pytorch_face_landmark/blob/master/test_camera_light_onnx.py#L143
# Using MobileNetV2 (56×56)
class MobileNetV2FaceLandmark():
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

    def preprocess_img(self, img):
        # Img from detection is already in RGB
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # ??? RGB

        img = cv2.resize(img, (56, 56))  # Based on MobileNetV2
        img = img.astype(np.float32) / 255.0  # ???? [0,1]
        img = np.transpose(img, (2, 0, 1))  # ?????? (H, W, C) -> (C, H, W)
        # taken from https://github.com/FaceONNX/FaceONNX/blob/main/netstandard/FaceONNX.Addons/face/classes/FaceAgeEstimator.cs#L72
        for i, v in enumerate([0.485, 0.456, 0.406]):
            img[i, ...] = img[i, ...] - v
        for i, v in enumerate([0.229, 0.224, 0.225]):
            img[i, ...] = img[i, ...] / v
        input_data = np.expand_dims(img, axis=0)
        return input_data

    def forward(self, img, bbox):
        (x1, y1, x2, y2) = bbox
        w = x2 - x1
        h = y2 - y1

        input_img = self.preprocess_img(img)
        ort_inputs = {self.session.get_inputs()[0].name: input_img}
        ort_outputs = self.session.run(None, ort_inputs)
        landmarks = ort_outputs[0]
        landmarks = landmarks.reshape(-1, 2)
        landmarks = [landmarks[landmark_idx] for landmark_idx in LANDMARK_IDX_68_TO_5]

        reprojected_landmarks = []
        for landmark in landmarks:
            x = landmark[0] * w
            y = landmark[1] * h
            reprojected_landmarks.append((x, y))
        return reprojected_landmarks
