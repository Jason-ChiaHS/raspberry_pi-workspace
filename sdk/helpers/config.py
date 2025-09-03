import typing as t
from argparse import Namespace
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


@dataclass
class GenerateArtifactsConfig:
    profiling: bool = False
    output_tensor_and_image: bool = False
    # TODO: Maybe there is a better way, but the default is generated and used by Cam and BasePipeline
    artifact_directory: str = f"./artifact-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


@dataclass
class CamConfig:
    model_path: str = "./demos/gaze/models/nngaze.rpk"
    width: int = 2028
    height: int = 1520
    buffer_count: int = 2
    inference_data_queue_limit: int = 10
    controls: t.Optional[dict] = None  # For Camera Controls from raspi-cam-svr


# For scripts
@dataclass
class ShowResultConfig:
    cv2_show_window: bool = False
    cv2_window_resize_width: int = 960
    cv2_window_resize_height: int = 720


@dataclass
class UploadMetadataConfig:
    client_server_url: str = "http://localhost:8010"
    b64_resize_width: int = 1920
    b64_resize_height: int = 1080
    send_frame: bool = True


@dataclass
class UploadTriggerMetadataConfig:
    client_server_url: str = "http://localhost:8010"
    b64_resize_width: int = 1920
    b64_resize_height: int = 1080
    send_frame: bool = True


@dataclass()
class Config:
    log_level: int = 3
    single_thread: bool = False
    profiling: bool = False
    generate_artifacts: GenerateArtifactsConfig = field(
        default_factory=GenerateArtifactsConfig
    )
    cam: CamConfig = field(default_factory=CamConfig)
    show_result: ShowResultConfig = field(default_factory=ShowResultConfig)
    upload_metadata: UploadMetadataConfig = field(default_factory=UploadMetadataConfig)
    upload_trigger_metadata: UploadTriggerMetadataConfig = field(
        default_factory=UploadTriggerMetadataConfig
    )

    # To allow passing of a string to the demo DemoScript start function
    # and enable it to run differenct scripts
    custom_script: t.Optional[str] = None


def load_config(args: Namespace) -> Config:
    CONFIG_PATH = Path("./config.yaml")
    config = OmegaConf.to_container(OmegaConf.structured(Config()))
    if CONFIG_PATH.exists():
        config = OmegaConf.merge(config, OmegaConf.load(CONFIG_PATH))
    args_config = OmegaConf.from_dotlist(
        [f"{k}={v}" for k, v in vars(args).items() if v is not None]
    )
    config = OmegaConf.merge(config, args_config)
    demo_pipeline_config_path = (
        Path(config.demo_pipeline_init_path).parent / "config.yaml"
    )
    if demo_pipeline_config_path.exists():
        demo_pipeline_config = OmegaConf.load(demo_pipeline_config_path)
        config = OmegaConf.merge(config, demo_pipeline_config)
    # So CLI args have the highest predence
    config = OmegaConf.merge(config, args_config)
    return config


def load_demo_config(config_path: Path) -> DictConfig:
    """
    Creates the base config, then loads and merges the given demo config if it exists
    """
    config = OmegaConf.structured(Config())
    if config_path.exists():
        config = OmegaConf.to_container(config)
        config = OmegaConf.merge(config, OmegaConf.load(config_path))
    return config
