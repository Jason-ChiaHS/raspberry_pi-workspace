"""
For other models we need to load
"""

import cv2
import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from . import logger


class BaseORT:
    def __init__(self, model_path):
        self.ort_session = ort.InferenceSession(model_path)

        # for input in self.ort_session.get_inputs():
        #     logger.info(f"{input.name}: {input.shape}, {input.type}")

        # for output in self.ort_session.get_outputs():
        #     logger.info(f"{output.name}: {output.shape}, {output.type}")


class MobileNet(BaseORT):
    def __init__(self, model_path="./models/mobilenetv4small.onnx"):
        super().__init__(model_path)
        inputs = self.ort_session.get_inputs()[0]
        self.model_width = inputs.shape[2]
        self.model_height = inputs.shape[3]
        logger.info(
            f"model_width: {self.model_width}, model_height: {self.model_height}"
        )

        self.start_age = 0
        self.end_age = 81

    def run_inference(self, image):
        """
        image: cv2 numpy image in RGB
        """
        img = image
        img = cv2.resize(img, (self.model_width, self.model_height))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        # Normalize
        for i, v in enumerate([0.485, 0.456, 0.406]):
            img[i, ...] = img[i, ...] - v
        for i, v in enumerate([0.229, 0.224, 0.225]):
            img[i, ...] = img[i, ...] / v
        input_data = np.expand_dims(img, axis=0)
        ort_inputs = {self.ort_session.get_inputs()[0].name: input_data}
        ort_outputs = self.ort_session.run(None, ort_inputs)

        output = ort_outputs[0]
        output = torch.from_numpy(output).cpu()
        m = nn.Softmax(dim=1)
        output = m(output)
        a = torch.arange(self.start_age, self.end_age + 1, dtype=torch.float32).cpu()
        mean = (output * a).sum(1, keepdim=True).cpu().data.numpy()
        predicted_age = np.around(mean)[0][0]
        # logger.info(f"predicted_age: {predicted_age}")

        return predicted_age  # converts to a single int at the end
