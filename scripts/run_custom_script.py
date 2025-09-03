import argparse
import typing as t

from sdk.base_pipeline import BaseScript
from sdk.helpers.argparse import add_demo_arg, add_log_level_arg
from sdk.helpers.config import load_demo_config, Config
from sdk.helpers.importer import import_module
from sdk.helpers.logger import logger, setup_level




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_log_level_arg(parser)
    demos = add_demo_arg(parser)
    parser.add_argument(
        "--custom_script",
        type=str,
        required=False
    )
    args = parser.parse_args()

    config = load_demo_config(demos[args.demo]["config"])
    config = t.cast(Config, config)
    config.custom_script = config.custom_script if args.custom_script is None else args.custom_script
    setup_level(config.log_level if args.log_level is None else args.log_level)
    logger.info(f"Loaded Config: {config}")

    demo_pipeline_module = import_module("demo_pipeline", demos[args.demo]["init"])
    demo_script = demo_pipeline_module.DemoScript(config)
    demo_script = t.cast(BaseScript, demo_script)
    demo_script.start()
