import logging
import sys
import typing as t
from pathlib import Path

# Adds the current
sys.path.append(Path(__file__).parent.absolute().as_posix())

from sdk.base_pipeline import BasePipeline  # noqa: F401 - Used in import later on
from sdk.helpers.logger import logger

logger = t.cast(logging.Logger, logger)

from .demo_pipeline import DemoPipeline
