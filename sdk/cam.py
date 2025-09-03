import queue
import threading
import time
import typing as t
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf
from picamera2 import CompletedRequest, Picamera2
from picamera2.devices import IMX500

from .cam_controls import read_camera_controls
from .helpers.artifact_generator import ArtifactGenerator, ProfilerArtifactGenerator
from .helpers.config import Config
from .helpers.logger import logger


class Cam:
    # take in params from argparse
    def __init__(self, config: Config):
        self.config = config
        self.model_path = self.config.cam.model_path
        self.profiling = self.config.profiling
        self.width = self.config.cam.width
        self.height = self.config.cam.height

        self.profiler = CamProfiler(self.config)

        self.imx500 = IMX500(self.model_path)
        self.imx500.show_network_fw_progress_bar()

        (self.model_width, self.model_height) = self.imx500.get_input_size()

        self.picam2 = Picamera2(self.imx500.camera_num)
        self.picam_config = self.picam2.create_video_configuration(
            {"size": (self.width, self.height)},
            buffer_count=self.config.cam.buffer_count,
        )
        self.picam2.configure(self.picam_config)
        # Ignoring camera controls from the demo config.yaml
        # self.picam2.set_controls(self.config_to_camera_controls())
        self.picam2.set_controls(read_camera_controls())

        self.picam2.pre_callback = self.camera_callback
        if self.config.single_thread:
            self.inference_data_queue = queue.Queue(maxsize=0)
        else:
            self.inference_data_queue = queue.Queue()

        if self.config.generate_artifacts.output_tensor_and_image:
            self.output_tensor_and_image_artifact_generator = (
                CamOutputTensorAndImageArtifactGenerator(
                    Path(self.config.generate_artifacts.artifact_directory)
                )
            )
        self.running = False

    def config_to_camera_controls(self) -> dict:
        if self.config.cam.controls is None:
            return {}
        camera_controls = OmegaConf.to_object(self.config.cam.controls)
        for k in camera_controls.keys():
            if isinstance(camera_controls[k], list):
                camera_controls[k] = tuple(camera_controls[k])
        return camera_controls

    def start_with_inference_function(self, inference_function):
        """
        inference_function being called everything np_outputs and img is ready from the camera
        """
        self.picam2.start()
        time.sleep(
            2
        )  # Wait for camera capture to run, smoother process if the camera buffers are already populated

        if not self.config.single_thread:
            self.process_inference_data_queue_thread = threading.Thread(
                target=self.process_inference_data_queue,
                args=(inference_function,),
                daemon=True,
            )
            self.process_inference_data_queue_thread.start()
        # Passing inference function in here for single threaded operation
        self.picam_loop_thread = threading.Thread(
            target=self.picam_loop, args=(inference_function,), daemon=True
        )
        self.running = True
        self.picam_loop_thread.start()

    def picam_loop(self, inference_function):
        while self.running:
            request = t.cast(CompletedRequest, self.picam2.capture_request())
            metadata = request.get_metadata()
            frame_time = time.monotonic()
            self.profiler.profile_camera_frame_time(frame_time)
            if metadata is not None:
                np_outputs = self.imx500.get_outputs(metadata)
                kpi = self.imx500.get_kpi_info(metadata)
                if np_outputs is not None:
                    logger.debug(kpi)
                    if len(np_outputs) > 0:
                        img = request.make_array("main").copy()
                        self.profiler.profile_inference_frame_time(frame_time)
                        inference_data = (np_outputs, img, frame_time, metadata)
                        if self.config.single_thread:
                            self.process_inference_data(
                                inference_data, inference_function
                            )
                        else:
                            self.inference_data_queue.put(inference_data)
            request.release()

    def stop(self):
        self.running = False
        self.picam_loop_thread.join()
        self.picam2.stop()
        self.picam2.close()
        if not self.config.single_thread:
            self.inference_data_queue.put(None)
            self.process_inference_data_queue_thread.join()

    def process_inference_data_queue(self, inference_function):
        while (inference_data := self.inference_data_queue.get()) is not None:
            if (
                self.inference_data_queue.qsize()
                > self.config.cam.inference_data_queue_limit
            ):
                # Warns that the inference function is running too slowly
                logger.warning(
                    f"SKIPPING Inference Frames as inference function is too slow, qsize: {self.inference_data_queue.qsize()}"
                )
                # Skips oldest inference_data in queue based on set limit
                continue
            self.process_inference_data(inference_data, inference_function)

    def process_inference_data(self, inference_data, inference_function):
        np_outputs, img, frame_time, metadata = (
            inference_data[0],
            inference_data[1],
            inference_data[2],
            inference_data[3],
        )
        start = time.monotonic()
        if self.config.generate_artifacts.output_tensor_and_image:
            self.output_tensor_and_image_artifact_generator.save_output_tensor_and_image(
                np_outputs, img.copy(), frame_time
            )
        inference_function(np_outputs, img, frame_time, metadata)
        end = time.monotonic()
        self.profiler.profile_inference_function_start_end(frame_time, start, end)

    def camera_callback(self, request: CompletedRequest):
        pass


class CamProfiler:
    def __init__(self, config: Config):
        self.max_len = 30  # Sensible 30 since camera fps being 30 and inference fps being around 10
        self.config = config
        self.camera_frame_times = deque(maxlen=self.max_len)
        self.camera_frame_counter = 0
        self.inference_frame_times = deque(maxlen=self.max_len)
        self.inference_frame_counter = 0
        self.actual_frame_times = deque(maxlen=self.max_len)
        self.actual_frame_counter = 0
        if self.config.generate_artifacts.profiling:
            self.artifact_generator = CamProfilerArtifactGenerator(
                Path(self.config.generate_artifacts.artifact_directory), "cam"
            )

    def profile_camera_frame_time(self, frame_time: float):
        if not self.config.profiling:
            return

        self.camera_frame_times.append(frame_time)
        if self.config.generate_artifacts.profiling:
            self.artifact_generator.queue_insert_statement_with_data(
                "INSERT INTO camera_frame_times VALUES(?)", [(frame_time,)]
            )
        if len(self.camera_frame_times) > 2 and self.camera_frame_counter == 0:
            logger.info(
                f"Camera FPS: {(len(self.camera_frame_times) - 1) / (self.camera_frame_times[-1] - self.camera_frame_times[0])}"
            )
        self.camera_frame_counter += 1
        self.camera_frame_counter %= self.max_len

    def profile_inference_frame_time(self, frame_time: float):
        if not self.config.profiling:
            return

        self.inference_frame_times.append(frame_time)
        if self.config.generate_artifacts.profiling:
            self.artifact_generator.queue_insert_statement_with_data(
                "INSERT INTO inference_frame_times VALUES(?)", [(frame_time,)]
            )
        if len(self.inference_frame_times) > 2 and self.inference_frame_counter == 0:
            logger.info(
                f"Camera Inference FPS: {(len(self.inference_frame_times) - 1) / (self.inference_frame_times[-1] - self.inference_frame_times[0])}"
            )
        self.inference_frame_counter += 1
        self.inference_frame_counter %= self.max_len

    def profile_inference_function_start_end(
        self, inference_frame_time: float, start: float, end: float
    ):
        if not self.config.profiling:
            return

        # Calculating actual frame times (And running the inference function)
        self.actual_frame_times.append(inference_frame_time)
        if self.config.generate_artifacts.profiling:
            self.artifact_generator.queue_insert_statement_with_data(
                "INSERT INTO actual_frame_times VALUES(?)", [(inference_frame_time,)]
            )
        if len(self.actual_frame_times) > 2 and self.actual_frame_counter == 0:
            logger.info(
                f"Actual FPS: {(len(self.actual_frame_times) - 1) / (self.actual_frame_times[-1] - self.actual_frame_times[0])}"
            )
        self.actual_frame_counter += 1
        self.actual_frame_counter %= self.max_len

        logger.debug(f"Inference Function time taken: {end - start}")
        if self.config.generate_artifacts.profiling:
            self.artifact_generator.queue_insert_statement_with_data(
                "INSERT INTO inference_function_times VALUES(?, ?, ?)",
                [(inference_frame_time, start, end)],
            )


class CamProfilerArtifactGenerator(ProfilerArtifactGenerator):
    def create_table(self, cur):
        cur.execute("CREATE TABLE camera_frame_times(time FLOAT NOT NULL)")
        cur.execute("CREATE TABLE inference_frame_times(time FLOAT NOT NULL)")
        cur.execute("CREATE TABLE actual_frame_times(time FLOAT NOT NULL)")
        cur.execute(
            "CREATE TABLE inference_function_times(inference_frame_time FLOAT NOT NULL, start FLOAT NOT NULL, end FLOAT NOT NULL)"
        )


class CamOutputTensorAndImageArtifactGenerator(ArtifactGenerator):
    def save_output_tensor_and_image(
        self, np_outputs: t.List[np.ndarray], img, inference_frame_time: float
    ):
        artifact_folder_path = self.artifact_folder_path / str(inference_frame_time)
        artifact_folder_path.mkdir(
            parents=True, exist_ok=True
        )  # We might create this in pipelines to debug
        # Saving image
        img_path = artifact_folder_path / "original.png"
        cv2.imwrite(img_path.as_posix(), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # Saving tensors
        tensor_folder_path = artifact_folder_path / "output_tensors"
        tensor_folder_path.mkdir(parents=True)
        for i, tensor in enumerate(np_outputs):
            tensor_path = tensor_folder_path / f"{i}.npy"
            np.save(tensor_path, tensor)
