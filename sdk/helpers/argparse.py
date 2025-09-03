from argparse import ArgumentParser
from pathlib import Path


def add_log_level_arg(parser: ArgumentParser):
    """
    Adds the --log_level argument to the parser
    """
    parser.add_argument(
        "--log_level", type=int, help="Minimum log level for message to be logged"
    )


def add_demo_arg(parser: ArgumentParser) -> dict:
    """
    Adds the --demo argument to the parser
    The possible options for demo is generated from the demos in the `./demos` folder
    Returns a mapping of the givens choices to their paths
    """
    demos_path = Path("./demos")
    demos = {}
    for demo_path in demos_path.iterdir():
        if not demo_path.is_dir():
            continue  # Skip non demo dirs
        demo_name = demo_path.stem
        demos[demo_name] = {
            "init": demo_path / "__init__.py",
            "config": demo_path / "config.yaml",
        }
    parser.add_argument(
        "--demo",
        type=str,
        default="template" if "template" in demos else demos.keys()[0],
        choices=demos.keys(),
        help="Demo you want to run the script with",
    )
    return demos
