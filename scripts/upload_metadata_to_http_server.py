import argparse
import typing as t

import requests

from sdk.base_pipeline import BasePipeline
from sdk.cam import Cam
from sdk.helpers.argparse import add_demo_arg, add_log_level_arg
from sdk.helpers.config import load_demo_config
from sdk.helpers.importer import import_module
from sdk.helpers.logger import logger, setup_level


def inference_function(np_outputs, img, frame_time: float, cam_metadata):
    (frame, metadata) = demo_pipeline.upload_metadata(
        np_outputs, img, frame_time, cam_metadata
    )
    image_url = f"{config.upload_metadata.client_server_url}/image/{frame_time}"
    meta_url = f"{config.upload_metadata.client_server_url}/meta/{frame_time}"
    try:
        session.put(image_url, data=frame, timeout=2)
        session.put(meta_url, json=metadata, timeout=2)
    except:
        logger.info("Failed to send data through HTTP")
        pass


if __name__ == "__main__":
    session = requests.session()
    parser = argparse.ArgumentParser()
    add_log_level_arg(parser)
    demos = add_demo_arg(parser)
    args = parser.parse_args()

    config = load_demo_config(demos[args.demo]["config"])
    setup_level(config.log_level if args.log_level is None else args.log_level)

    cam = Cam(config)

    demo_pipeline_module = import_module("demo_pipeline", demos[args.demo]["init"])
    demo_pipeline = demo_pipeline_module.DemoPipeline(config, cam)
    demo_pipeline = t.cast(BasePipeline, demo_pipeline)
    cam.start_with_inference_function(inference_function)
    cam.picam_loop_thread.join()
