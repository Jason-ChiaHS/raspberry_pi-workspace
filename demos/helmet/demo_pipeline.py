from dataclasses import dataclass, field

import cv2
import helmet_utils as utils

from sdk.helpers.config import Config

from . import BasePipeline


@dataclass
class Helmet:
    # det box with score lower than this will not be drawn on img
    score_threshold: float = 0.4

    # Name of classes, this will be used to label the det box
    classes: dict[int, str] = field(
        default_factory=lambda: {
            0: "helmet",
            1: "head_with_helmet",
            2: "person_with_helmet",
            3: "head",
            4: "person_no_helmet",
            5: "face",
        }
    )
    default_color: list[int] = field(default_factory=lambda: [0, 0, 0])

    # Set the color of box and text for each class in BGR color format,
    # if no color set, default color (config.helmet.default_color) will be used
    # set it as empty list [] if not specifying color
    colors: dict[int, list[int]] = field(
        default_factory=lambda: {
            0: [224, 128, 245],
            1: [],
            2: [0, 255, 0],
            3: [],
            4: [0, 0, 255],
            5: [],
        }
    )

    # det box with label appearing in skip_label will not be drawn on img
    # set it as empty list [] if drawing all labels
    skip_label: list[int] = field(default_factory=lambda: [1, 3, 5])


# Implementing the AdditionalDemoConfig is not strictly needed
# Its purpose is to set default values and help with typing of any added vars
@dataclass
class AdditionalDemoConfig(Config):
    helmet: Helmet = field(default_factory=Helmet)


# Override the methods below, referring to their documentation
# Refer to sdk/base_pipeline.py BasePipeline for how they are used in the respective scripts
class DemoPipeline(BasePipeline):
    def __init__(self, config, cam):
        # Just extend __init__
        super().__init__(config, cam)
        self.config: AdditionalDemoConfig = self.merge_demo_pipeline_config(
            AdditionalDemoConfig()
        )

        self.score_threshold = self.config.helmet.score_threshold
        self.width_scale = self.config.cam.width / self.model_width
        self.height_scale = self.config.cam.height / self.model_height
        self.skip_label = self.config.helmet.skip_label
        self.label_dict = self.config.helmet.classes

        self.colors_dict = utils.get_color_dict(
            self.config.helmet.colors, self.config.helmet.default_color, self.label_dict
        )

    def post_processing(self, inference_frame_time, np_outputs, img, cam_metadata):
        # Init list to store outputs to return
        det_list = []

        box_outputs, score_outputs, label_outputs = np_outputs[:3]

        for i in range(len(label_outputs)):
            # Skip boxes with score lower than threshold or with labels in self.skip_label
            if (score_outputs[i] < self.score_threshold) or (
                label_outputs[i] in self.skip_label
            ):
                continue

            det_list.append(
                {
                    "box": utils.scale_box(
                        box_outputs[i], self.width_scale, self.height_scale
                    ),
                    "score": score_outputs[i],
                    "label": label_outputs[i],
                }
            )

        return det_list

    def draw_metadata(self, inference_frame_time, metadata, img):
        for data in metadata:
            label = int(data["label"])
            color = self.colors_dict[label]
            x1, y1, x2, y2 = data["box"]
            img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            img = cv2.putText(
                img,
                f"{data['score']:.2f}_{self.label_dict[label]}",
                (int(x1), int(y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )
        return img

    def serialize_metadata(self, inference_frame_time, metadata):
        serialized_metadata = []
        for det in metadata:
            serialized_metadata.append(
                {
                    "box": [box.tolist() for box in det["box"]],
                    "score": det["score"].tolist(),
                    "label": det["label"].tolist(),
                }
            )
        return serialized_metadata
