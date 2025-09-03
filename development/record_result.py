import argparse
from datetime import datetime
from pathlib import Path
import typing as t

from sdk.base_pipeline import BasePipeline
from sdk.cam import Cam
from sdk.helpers.argparse import add_demo_arg, add_log_level_arg
from sdk.helpers.config import load_demo_config, Config
from sdk.helpers.importer import import_module
from sdk.helpers.logger import logger, setup_level
import cv2


def inference_function(np_outputs, img, frame_time: float, cam_metadata):
    metadata = demo_pipeline.profile_post_processing(frame_time, np_outputs, img, cam_metadata)
    img = demo_pipeline.profile_draw_metadata(frame_time, metadata, img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    out.write(img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_log_level_arg(parser)
    demos = add_demo_arg(parser)
    args = parser.parse_args()

    config = load_demo_config(demos[args.demo]["config"])
    config = t.cast(Config, config)
    setup_level(config.log_level if args.log_level is None else args.log_level)
    logger.info(f"Loaded Config: {config}")

    record_result_path = Path(f"./record_result-{args.demo}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    record_result_path.mkdir(parents=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also try 'XVID' or 'MJPG'
    out = cv2.VideoWriter((record_result_path /'output_video.mp4').as_posix(), fourcc, 10, (config.cam.width, config.cam.height))


    cam = Cam(config)

    demo_pipeline_module = import_module("demo_pipeline", demos[args.demo]["init"])
    demo_pipeline = demo_pipeline_module.DemoPipeline(config, cam)
    demo_pipeline = t.cast(BasePipeline, demo_pipeline)
    cam.start_with_inference_function(inference_function)
    try:
        cam.picam_loop_thread.join()
    except KeyboardInterrupt:
        out.release()
