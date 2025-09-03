import argparse
import typing as t

import requests

from sdk.base_pipeline import BasePipeline
from sdk.cam import Cam
from sdk.helpers.argparse import add_demo_arg, add_log_level_arg
from sdk.helpers.config import Config, load_demo_config
from sdk.helpers.importer import import_module
from sdk.helpers.logger import logger, setup_level


def inference_function(np_outputs, img, frame_time: float, cam_metadata):
    outputs = demo_pipeline.profile_post_processing(
        frame_time, np_outputs, img, cam_metadata
    )
    demo_pipeline.profile_update_trigger_condition(frame_time, outputs, img)
    if demo_pipeline.trigger_condition:
        (frame, metadata) = (
            demo_pipeline.b64_encode_image_with_config(
                img,
                config.upload_trigger_metadata.b64_resize_width,
                config.upload_trigger_metadata.b64_resize_height,
                config.upload_trigger_metadata.send_frame,
            ),
            demo_pipeline.serialize_trigger_metadata(frame_time, outputs),
        )
        demo_pipeline.reset_trigger_condition()

        image_url = (
            f"{config.upload_metadata.client_server_url}/trigger/image/{frame_time}"
        )
        meta_url = (
            f"{config.upload_metadata.client_server_url}/trigger/meta/{frame_time}"
        )
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
    config = t.cast(Config, config)
    setup_level(config.log_level if args.log_level is None else args.log_level)

    cam = Cam(config)

    demo_pipeline_module = import_module("demo_pipeline", demos[args.demo]["init"])
    demo_pipeline = demo_pipeline_module.DemoPipeline(config, cam)
    demo_pipeline = t.cast(BasePipeline, demo_pipeline)
    demo_pipeline.setup_trigger_condition()
    cam.start_with_inference_function(inference_function)
    cam.picam_loop_thread.join()
