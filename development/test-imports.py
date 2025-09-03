import argparse
import typing as t
from pathlib import Path

from sdk.base_pipeline import BasePipeline
from sdk.cam import Cam
from sdk.helpers.config import load_config
from sdk.helpers.importer import import_module

namespace = argparse.Namespace(
    demo_pipeline_init_path="./demos/helmet/__init__.py",
)
config = load_config(namespace)
cam = Cam(config)
demo_pipeline_module = import_module(
    "demo_pipeline", Path(config.demo_pipeline_init_path)
)
demo_pipeline = demo_pipeline_module.DemoPipeline(config, cam)
demo_pipeline = t.cast(BasePipeline, demo_pipeline)
cam.picam2.close()

namespace = argparse.Namespace(
    demo_pipeline_init_path="./demos/gaze/__init__.py",
)
config = load_config(namespace)
cam = Cam(config)
demo_pipeline_module = import_module(
    "demo_pipeline", Path(config.demo_pipeline_init_path)
)
demo_pipeline = demo_pipeline_module.DemoPipeline(config, cam)
demo_pipeline = t.cast(BasePipeline, demo_pipeline)
cam.picam2.close()
