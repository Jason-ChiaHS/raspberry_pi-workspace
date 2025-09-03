from pathlib import Path

import yaml

camera_control_path = Path("./camera_controls.yaml")


def read_camera_controls() -> dict:
    """
    Reads the defaults camera_control_path .yaml file and tries to load it
    """
    if not camera_control_path.exists():
        return {}
    camera_controls = yaml.safe_load(camera_control_path.read_text())
    for k, v in camera_controls.items():
        nv = v
        if isinstance(nv, list):
            nv = tuple(nv)
        camera_controls[k] = nv
    return camera_controls


def save_camera_controls(camera_controls: dict):
    """
    Given the new camera_controls dict, save it to the camera_control_path accounting for tuples
    """
    new_camera_controls = {}
    for k, v in camera_controls.items():
        nv = v
        if isinstance(nv, tuple):
            nv = list(nv)
        new_camera_controls[k] = nv
        camera_control_path.write_text(
            yaml.dump(new_camera_controls, default_flow_style=False)
        )
