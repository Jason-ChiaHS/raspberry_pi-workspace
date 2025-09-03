import sys
import logging
import threading
import time
import typing as t
from picamera2 import CompletedRequest, Picamera2
from picamera2.devices import IMX500


def setup_level(level: int):
    level_maping = {
        4: logging.CRITICAL,
        3: logging.ERROR,
        2: logging.WARNING,
        1: logging.INFO,
        0: logging.DEBUG,
    }
    logging_level = level_maping.get(level, logging.ERROR)
    logger.setLevel(logging_level)
    stdout_handler.setLevel(logging_level)


# Setting up sdk logger
logger = logging.getLogger("imx500-sdk")
logger.setLevel(logging.ERROR)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.ERROR)
formatter = logging.Formatter(
    fmt="[%(levelname)s] %(asctime)s|%(module)s.%(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)


class Cam:
    # take in params from argparse
    def __init__(self, model_path, width=2028, height=1520, buffer_count=2):
        self.model_path = model_path
        self.width = width
        self.height = height

        self.imx500 = IMX500(self.model_path)
        self.imx500.show_network_fw_progress_bar()

        (self.model_width, self.model_height) = self.imx500.get_input_size()

        self.picam2 = Picamera2(self.imx500.camera_num)
        self.picam_config = self.picam2.create_video_configuration(
            {"size": (self.width, self.height)},
            buffer_count=2,
        )
        self.picam2.configure(self.picam_config)

        # Not using this as it has more performance implications
        # self.picam2.pre_callback = self.camera_callback

    def start(self):
        self.picam2.start()
        # Wait for camera capture to run, smoother process if the camera buffers are already populated
        time.sleep(2)

        # Using thread instead of callback
        # self.picam_loop_thread = threading.Thread(
        #     target=self.picam_loop, daemon=True
        # )
        self.running = True
        # self.picam_loop_thread.start()

    def picam_loop(self):
        while self.running:
            request = t.cast(CompletedRequest, self.picam2.capture_request())
            metadata = request.get_metadata()
            if metadata is not None:
                np_outputs = self.imx500.get_outputs(metadata)
                kpi = self.imx500.get_kpi_info(metadata)
                if np_outputs is not None and len(np_outputs) > 0:
                    print(np_outputs[0].shape)
                    img = request.make_array("main").copy()
                    # kpi should be available when output tensors are available
                    dnn_runtime, dsp_runtime = kpi
                    # In ms
                    logger.info(
                        f"dnn_runtime: {dnn_runtime}, dsp_runtime: {dsp_runtime}"
                    )
            request.release()


# Run with python self_contained_benchmark.py
if __name__ == "__main__":
    setup_level(1)
    # CHANGE: replace with path to model
    cam = Cam("./demos/gaze-v2/models/network15.rpk")
    cam.start()
    cam.picam_loop()
    # Using thread instead
    # cam.picam_loop_thread.join()
