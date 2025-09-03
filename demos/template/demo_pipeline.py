import typing as t
from dataclasses import dataclass

from sdk.helpers.config import Config

from sdk.base_pipeline import BasePipeline, BaseScript


# Implementing the AdditionalDemoConfig is not strictly needed
# Its purpose is to set default values and help with typing of any added vars
@dataclass
class AdditionalDemoConfig(Config):
    pass

class DemoScript(BaseScript):
    def __init__(self, config):
        # Just extend __init__
        super().__init__(config)
        self.config: AdditionalDemoConfig = self.merge_demo_pipeline_config(
            AdditionalDemoConfig()
        )

    def start(self):
        print("Template demo script")
        return super().start()


# Override the methods below, referring to their documentation
# Refer to sdk/base_pipeline.py BasePipeline for how they are used in the respective scripts
class DemoPipeline(BasePipeline):
    def __init__(self, config, cam):
        # Just extend __init__
        super().__init__(config, cam)
        self.config: AdditionalDemoConfig = self.merge_demo_pipeline_config(
            AdditionalDemoConfig()
        )

    def post_processing(self, inference_frame_time, np_outputs, img, cam_metadata):
        return super().post_processing(inference_frame_time, np_outputs, img)

    def draw_metadata(self, inference_frame_time, metadata, img):
        return super().draw_metadata(inference_frame_time, metadata, img)

    def serialize_metadata(self, inference_frame_time, metadata):
        return super().serialize_metadata(inference_frame_time, metadata)
