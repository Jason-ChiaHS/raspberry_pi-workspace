import argparse
import queue
import sqlite3
import threading
import time
import typing as t
from datetime import datetime
from pathlib import Path

import cv2
from picamera2.request import CompletedRequest

from sdk.cam import Cam
from sdk.helpers.artifact_generator import Profiler
from sdk.helpers.config import ConfigLoader
from sdk.helpers.logger import logger, setup_level

CONFIG_PATH = "./config.toml"
SCRIPT_NAME = Path(__file__).stem


def handle_demo_pipeline_jobs(jobs: queue.Queue):
    while (job := jobs.get()) is not None:
        np_outputs, img = job[0], job[1]


def cv2_show_img(img):
    cv2_window_resize_width = 1200
    cv2_window_resize_height = 900
    img = cv2.resize(
        img, (cv2_window_resize_width, cv2_window_resize_height)
    )  # Resize image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("cv2 image preview", img)
    cv2.waitKey(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    config = ConfigLoader(CONFIG_PATH, args)
    setup_level(config.log_level)

    cam = Cam(
        config.model_path,
        config.profiling,
        config.width,
        config.height,
        config.buffer_count,
    )
    cam.start()
    logger.info("Starting Cam")
    time.sleep(2)

    profiler = Profiler(Path(f"./{int(datetime.now().timestamp())}"))
    profiler.start()

    demo_pipeline_jobs = queue.Queue()
    handle_demo_pipeline_jobs_thread = threading.Thread(
        target=handle_demo_pipeline_jobs,
        args=(demo_pipeline_jobs,),
        daemon=True,
    )
    handle_demo_pipeline_jobs_thread.start()

    camera_frame_times = []
    inference_frame_times = []
    i = 0
    j = 0

    while True:
        camera_frame_times.append(time.monotonic())
        if len(camera_frame_times) > 30:
            camera_frame_times.pop(0)
        if len(camera_frame_times) > 2:
            if i == 0:
                logger.info(
                    f"Camera FPS: {len(camera_frame_times) / (camera_frame_times[-1] - camera_frame_times[0])}"
                )
        request = t.cast(CompletedRequest, cam.picam2.capture_request())
        metadata = request.get_metadata()
        if metadata is not None:
            np_outputs = cam.imx500.get_outputs(metadata)
            if np_outputs is not None:
                inference_frame_times.append(time.monotonic())
                if len(inference_frame_times) > 30:
                    inference_frame_times.pop(0)
                if len(inference_frame_times) > 2:
                    if j == 0:
                        logger.info(
                            f"Camera Inference FPS: {len(inference_frame_times) / (inference_frame_times[-1] - inference_frame_times[0])}"
                        )
                # logger.info(f"np_outputs: {np_outputs}")
                if len(np_outputs) > 0:
                    img = request.make_array("main")
                    demo_pipeline_jobs.put((np_outputs, img))
                    j += 1
                    j = j % 30
        i += 1
        i = i % 30
        request.release()
