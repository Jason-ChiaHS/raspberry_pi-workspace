import logging
import sys
import typing as t
from pathlib import Path

# Adds the current
sys.path.append(Path(__file__).parent.absolute().as_posix())

from .demo_pipeline import DemoPipeline, DemoScript
