import base64
import time
from pathlib import Path

import cv2
from omegaconf import OmegaConf

from .cam import Cam
from .helpers.artifact_generator import ProfilerArtifactGenerator
from .helpers.config import Config
from .helpers.logger import logger

SQLITE_FUNCTION_ID_MAPPING = {
    "post_processing": 0,
    "draw_metadata": 1,
    "serialize_metadata": 2,
    "update_trigger_condition": 3,
}
FUNCTION_ID_SQLITE_MAPPING = {v: k for k, v in SQLITE_FUNCTION_ID_MAPPING.items()}

class BaseScript:
    """
    This is to allow DemoScript to inherit from this and merge the demo's config and set the self.config
    When the start method is called the demo can run whatever it wants
    """
    def __init__(self, config: Config):
        self.config = config

    def merge_demo_pipeline_config(self, additional_demo_config_default):
        """
        Used in the overide for __init__ in the Demo Pipeline\n
        Will merge additional_demo_config_default with self.config\n
        This way, the config defined in the demo pipeline will definitely be available

        :param demo_pipeline_config_default: Default of the AdditionalDemoConfig
        :returns: Merged Config object
        """
        config = self.config
        additional_demo_config_default = OmegaConf.to_container(
            OmegaConf.structured(additional_demo_config_default)
        )
        config = OmegaConf.merge(additional_demo_config_default, config)
        return config

    def start(self):
        # Only called by the run_custom_script
        pass

class BasePipeline(BaseScript):
    def __init__(self, config: Config, cam: Cam):
        """
        EXTEND\n
        Place anything you would need to initialize here

        :param config_data: Dict, Taken from the ConfigLoader, to make configs available in the class
        """
        super().__init__(config)
        self.cam = cam
        self.model_width = self.cam.model_width
        self.model_height = self.cam.model_height
        self.profiler = PipelineProfiler(self.config)

    # SCRIPT
    def show_result(self, np_outputs, img, inference_frame_time, cam_metadata):
        """
        np_outputs: Tensor outputs from the rpk model
        img: Original cv2 image from the camera
        """
        # Recommended to follow general the pipeline below
        metadata = self.profile_post_processing(
            inference_frame_time, np_outputs, img, cam_metadata
        )
        img = self.profile_draw_metadata(inference_frame_time, metadata, img)
        self.show_cv2_window(img)

    # SCRIPT
    def upload_metadata(self, np_outputs, img, inference_frame_time, cam_metadata):
        """
        np_outputs: Tensor outputs from the rpk model
        img: Original cv2 image from the camera
        """
        # Recommended to follow general the pipeline below
        metadata = self.profile_post_processing(
            inference_frame_time, np_outputs, img, cam_metadata
        )
        return (
            self.b64_encode_image(img),
            self.profile_serialize_metadata(inference_frame_time, metadata),
        )

    # OVERRIDE
    def post_processing(self, inference_frame_time, np_outputs, img, cam_metadata):
        """
        EXTEND/OVERRIDE\n
        Any general post-processing and running of other models\n
        img will be the native image taken from the camera with width: self.config.cam.width and height: self.config.cam.height\n
        Use these values to scale your model outputs\n
        You can also access the imx500 model at self.cam.imx500\n
        Return metadata

        :param np_outputs: Tensor outputs from the rpk model
        :param img: Original cv2 image from the camera
        :return: metadata
        """
        return None

    def profile_post_processing(
        self, inference_frame_time, np_outputs, img, cam_metadata
    ):
        start = time.monotonic()
        metadata = self.post_processing(
            inference_frame_time, np_outputs, img, cam_metadata
        )
        end = time.monotonic()
        self.profiler.profile_pipeline_function_with_id(
            inference_frame_time,
            SQLITE_FUNCTION_ID_MAPPING["post_processing"],
            start,
            end,
        )
        return metadata

    # OVERRIDE
    def draw_metadata(self, inference_frame_time, metadata, img):
        """
        EXTEND/OVERRIDE\n
        Anything you want drawn onto the native img\n
        metadata is the same as the returned value from self.post_processing\n
        Return img to draw

        :param metadata: metadata from self.post_processing
        :param img: Original cv2 image from the camera
        :return: cv2 img to be shown
        """
        return img

    def profile_draw_metadata(self, inference_frame_time, metadata, img):
        start = time.monotonic()
        img = self.draw_metadata(inference_frame_time, metadata, img)
        end = time.monotonic()
        self.profiler.profile_pipeline_function_with_id(
            inference_frame_time,
            SQLITE_FUNCTION_ID_MAPPING["draw_metadata"],
            start,
            end,
        )
        return img

    def show_cv2_window(self, img):
        """
        img: Output img from process_img
        """
        cv2_show_window = self.config.show_result.cv2_show_window
        if cv2_show_window:
            cv2_window_resize_width = self.config.show_result.cv2_window_resize_width
            cv2_window_resize_height = self.config.show_result.cv2_window_resize_height
            img = cv2.resize(
                img, (cv2_window_resize_width, cv2_window_resize_height)
            )  # Resize image
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow("CV2 Show Result", img)
            cv2.waitKey(1)

    def b64_encode_image(self, img):
        """
        img: Output img from process_img

        Encodes the img to a base64 bytes
        """
        if not self.config.upload_metadata.send_frame:
            return b""
        b64_resize_width = self.config.upload_metadata.b64_resize_width
        b64_resize_height = self.config.upload_metadata.b64_resize_height
        img = cv2.resize(img, (b64_resize_width, b64_resize_height))  # Resize image
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _, encoded_image = cv2.imencode(".jpg", image)
        frame = base64.b64encode(encoded_image.tobytes())
        return frame

    def b64_encode_image_with_config(
        self, img, b64_resize_width, b64_resize_height, send_frame
    ):
        """
        img: Output img from process_img

        Encodes the img to a base64 bytes
        """
        if not send_frame:
            return b""
        img = cv2.resize(img, (b64_resize_width, b64_resize_height))  # Resize image
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _, encoded_image = cv2.imencode(".jpg", image)
        frame = base64.b64encode(encoded_image.tobytes())
        return frame

    # OVERRIDE
    def serialize_metadata(self, inference_frame_time, metadata):
        """
        EXTEND/OVERRIDE\n
        Converts the metadata output into a JSON serializable format\n
        Example being converting pytorch `Tensor` and numpy `ndarray` to serializable python list of floats

        :param metadata: From self.post_processing
        :return: JSON serializable metadata
        """
        return {}

    def profile_serialize_metadata(self, inference_frame_time, metadata):
        start = time.monotonic()
        serialized_metadata = self.serialize_metadata(inference_frame_time, metadata)
        end = time.monotonic()
        self.profiler.profile_pipeline_function_with_id(
            inference_frame_time,
            SQLITE_FUNCTION_ID_MAPPING["serialize_metadata"],
            start,
            end,
        )
        return serialized_metadata

    def setup_trigger_condition(self):
        """
        EXTEND/OVERRIDE\n
        Setups the self.trigger_condition and anything else
        """
        self.trigger_condition = False

    def update_trigger_condition(self, inference_frame_time, metadata, img):
        """
        EXTEND/OVERRIDE\n
        Function that runs on every frame after post-processing\n
        Should update the self.trigger_condition
        """
        pass

    def profile_update_trigger_condition(self, inference_frame_time, metadata, img):
        start = time.monotonic()
        self.update_trigger_condition(inference_frame_time, metadata, img)
        end = time.monotonic()
        self.profiler.profile_pipeline_function_with_id(
            inference_frame_time,
            SQLITE_FUNCTION_ID_MAPPING["update_trigger_condition"],
            start,
            end,
        )

    def reset_trigger_condition(self):
        """
        EXTEND/OVERRIDE\n
        Reset the self.trigger_condition and anything else\n
        Called after serialize_trigger_metadata
        """
        self.trigger_condition = False

    def serialize_trigger_metadata(self, inference_frame_time, metadata):
        """
        EXTEND/OVERRIDE\n
        Converts the metadata output into a JSON serializable format\n
        Runs when triggered on a frame
        """
        pass


class PipelineProfiler:
    def __init__(self, config: Config):
        self.config = config
        if self.config.generate_artifacts.profiling:
            self.artifact_generator = PipelineProfilerArtifactGenerator(
                Path(self.config.generate_artifacts.artifact_directory), "pipeline"
            )

    def profile_pipeline_function_with_id(
        self, inference_frame_time: float, function_id: int, start: float, end: float
    ):
        if not self.config.profiling:
            return
        logger.debug(
            f"Pipeline Function, {FUNCTION_ID_SQLITE_MAPPING[function_id]} took: {end - start}"
        )
        if self.config.generate_artifacts.profiling:
            self.artifact_generator.queue_insert_statement_with_data(
                "INSERT INTO demo_pipeline VALUES(?, ?, ?, ?)",
                [(inference_frame_time, function_id, start, end)],
            )


class PipelineProfilerArtifactGenerator(ProfilerArtifactGenerator):
    def create_table(self, cur):
        cur.execute(
            "CREATE TABLE demo_pipeline(inference_frame_time FLOAT NOT NULL, function_id INT NOT NULL, start FLOAT NOT NULL, end FLOAT NOT NULL)"
        )
