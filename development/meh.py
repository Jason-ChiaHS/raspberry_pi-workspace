import argparse
import time
import typing as t
from pathlib import Path

from sdk.base_pipeline import BasePipeline
from sdk.cam import Cam
from sdk.helpers.config import config_merge_namespace, load_config
from sdk.helpers.importer import import_module
from sdk.helpers.logger import setup_level


def inference_function(np_outputs, img, frame_time: float):
    demo_pipeline.show_result(np_outputs, img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cv2_show_window",
        action=argparse.BooleanOptionalAction,
        dest="show_result.cv2_show_window",
    )
    args = parser.parse_args()

    config = load_config()
    config = config_merge_namespace(config, args)
    setup_level(config.log_level)

    cam = Cam(config)

    demo_pipeline_module = import_module(
        "demo_pipeline", Path(config.demo_pipeline_init_path)
    )
    demo_pipeline = demo_pipeline_module.DemoPipeline(config, cam)
    demo_pipeline = t.cast(BasePipeline, demo_pipeline)
    cam.start_with_inference_function(inference_function)
