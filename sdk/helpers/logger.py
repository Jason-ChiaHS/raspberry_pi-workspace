import logging
import sys


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
