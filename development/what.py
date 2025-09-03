import argparse
import typing as t

from sdk.base_pipeline import BasePipeline
from sdk.cam import Cam
from sdk.helpers.argparse import add_demo_arg, add_log_level_arg
from sdk.helpers.config import Config, load_demo_config
from sdk.helpers.importer import import_module
from sdk.helpers.logger import logger, setup_level


def inference_function(np_outputs, img, frame_time: float, cam_metadata):
    metadata = demo_pipeline.profile_post_processing(
        frame_time, np_outputs, img, cam_metadata
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_log_level_arg(parser)
    demos = add_demo_arg(parser)
    parser.add_argument(
        "--cv2_show_window",
        action=argparse.BooleanOptionalAction,
    )
    args = parser.parse_args()

    config = load_demo_config(demos[args.demo]["config"])
    config = t.cast(Config, config)
    config.show_result.cv2_show_window = (
        config.show_result.cv2_show_window
        if args.cv2_show_window is None
        else args.cv2_show_window
    )
    setup_level(config.log_level if args.log_level is None else args.log_level)
    logger.info(f"Loaded Config: {config}")

    cam = Cam(config)

    demo_pipeline_module = import_module("demo_pipeline", demos[args.demo]["init"])
    demo_pipeline = demo_pipeline_module.DemoPipeline(config, cam)
    demo_pipeline = t.cast(BasePipeline, demo_pipeline)
    cam.start_with_inference_function(inference_function)
    cam.picam_loop_thread.join()
