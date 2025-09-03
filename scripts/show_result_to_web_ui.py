import argparse
import json
import queue
import threading
import typing as t
from pathlib import Path

from flask import Flask, abort, request, send_from_directory
from flask_socketio import SocketIO
from omegaconf import OmegaConf
from raspiCamSrv import create_app
from raspiCamSrv.camera_pi import Camera

from sdk.base_pipeline import BasePipeline
from sdk.cam import Cam
from sdk.helpers.config import load_config
from sdk.helpers.importer import import_module
from sdk.helpers.logger import setup_level

folder_path = Path(__file__).parent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_level", type=int, default=3)
    args = parser.parse_args()

    socketio_namespace = "/image_meta"
    socketio_image_meta_data_queue = queue.Queue()

    # config = load_config(args)
    setup_level(args.log_level)

    app = Flask(__name__)
    # socketio = SocketIO(async_mode="threading", cors_allowed_origins="*")
    socketio = SocketIO(cors_allowed_origins="*")

    @socketio.on("connect", namespace=socketio_namespace)
    def connect():
        pass

    def socketio_emit_image_meta():
        while True:
            (frame, metadata) = socketio_image_meta_data_queue.get()
            frame = t.cast(bytes, frame)
            socketio.emit(
                "recv_image_meta",
                {"frame": frame.decode("utf-8"), "metadata": metadata},
                namespace=socketio_namespace,
            )

    socketio_thread = threading.Thread(target=socketio_emit_image_meta, daemon=True)
    socketio_thread.start()

    cam_control_app = create_app()

    def run_cam_control_app(app: Flask):
        app.run(host="0.0.0.0", port=9001)

    cam_control_app_thread = threading.Thread(
        target=run_cam_control_app, args=(cam_control_app,), daemon=True
    )
    cam_control_app_thread.start()

    # Path for our main Svelte page
    @app.route("/")
    def base():
        return send_from_directory(
            (folder_path / "web_ui/build").as_posix(),
            "index.html",
        )

    # Path for all the static files (compiled JS/CSS, etc.)
    @app.route("/<path:path>")
    def home(path):
        return send_from_directory(
            (folder_path / "web_ui/build").as_posix(),
            path,
        )

    def get_demos_data():
        demos_path = Path("./demos")
        demos = {}
        for demo_path in demos_path.iterdir():
            if not demo_path.is_dir():
                continue  # Skip non demo dirs
            demo_name = demo_path.stem
            demos[demo_name] = {
                "name": demo_name.capitalize(),
                "demo_pipeline_init_path": (demo_path / "__init__.py").as_posix(),
                "config": demo_path / "config.yaml",
            }
        if "template" in demos:
            del demos["template"]
        return demos

    demos_data = get_demos_data()

    class DemoRunner:
        def __init__(self):
            self.cam: t.Optional[Cam] = None
            self.running_demo: t.Optional[str] = None

        def write_config(self, selected_demo, demo_config):
            demo_config_path = (
                Path(demos_data[selected_demo]["demo_pipeline_init_path"]).parent
                / "config.yaml"
            )
            demo_config_path.write_text(demo_config)

        def restart(self, selected_demo):
            Camera().stopCameraSystem()
            if self.cam is not None:
                self.cam.stop()
            # Load the demo_config
            namespace = argparse.Namespace(
                demo_pipeline_init_path=demos_data[selected_demo][
                    "demo_pipeline_init_path"
                ]
            )
            config = load_config(namespace)
            self.cam = Cam(config)
            demo_pipeline_module = import_module(
                "demo_pipeline", Path(config.demo_pipeline_init_path)
            )
            demo_pipeline = demo_pipeline_module.DemoPipeline(config, self.cam)
            demo_pipeline = t.cast(BasePipeline, demo_pipeline)

            def create_inference_function(demo_pipeline):
                def inference_function(
                    np_outputs, img, frame_time: float, cam_metadata
                ):
                    metadata = demo_pipeline.profile_post_processing(
                        frame_time, np_outputs, img, cam_metadata
                    )
                    frame = demo_pipeline.profile_draw_metadata(
                        frame_time, metadata, img
                    )
                    frame = demo_pipeline.b64_encode_image(frame)
                    metadata = demo_pipeline.profile_serialize_metadata(
                        frame_time, metadata
                    )
                    socketio_image_meta_data_queue.put((frame, metadata))

                return inference_function

            self.cam.start_with_inference_function(
                create_inference_function(demo_pipeline)
            )
            self.running_demo = selected_demo

        def stop(self):
            Camera().stopCameraSystem()
            if self.cam is not None:
                self.cam.stop()
            self.running_demo = None

        def restart_raspi_cam_srv(self):
            if self.cam is not None:
                self.cam.stop()
            Camera().startLiveStream()
            self.running_demo = None

    demo_runner = DemoRunner()

    @app.post("/api/start")
    def start_demo():
        demo_data = request.get_json()
        # Ensure both keys are in json
        if "value" not in demo_data or "config" not in demo_data:
            return abort(400)
        # Check the given config is valid
        try:
            OmegaConf.create(demo_data["config"])
        except Exception:
            return abort(400)
        demo_runner.write_config(demo_data["value"], demo_data["config"])
        demo_runner.restart(demo_data["value"])
        return ""

    @app.post("/api/start_demo")
    def start_demo_v2():
        demo_data = request.get_json()
        # Ensure both keys are in json
        demo_runner.restart(demo_data["value"])
        return ""

    @app.get("/api/running_demo")
    def get_running_demo():
        return (
            demo_runner.running_demo if demo_runner.running_demo is not None else "None"
        )

    @app.post("/api/stop")
    def stop_demo():
        demo_runner.stop()
        return ""

    @app.post("/api/start_raspi_cam_srv")
    def start_raspi_cam_srv():
        demo_runner.restart_raspi_cam_srv()
        return ""

    @app.get("/api/demos")
    def get_demos():
        return [{"value": k, "name": v["name"]} for k, v in demos_data.items()]

    @app.post("/api/demo_config/get")
    def get_demo_config():
        selected_demo = request.get_data(as_text=True)
        demo = demos_data.get(selected_demo)
        if demo is None:
            abort(400)
        return demo["config"].read_text()

    @app.post("/api/demo_config_spec/get")
    def get_demo_config_spec():
        selected_demo = request.get_data(as_text=True)
        demo = demos_data.get(selected_demo)
        if demo is None:
            abort(400)
        demo_config = OmegaConf.to_container(OmegaConf.load(demo["config"]))

        def to_demo_config_spec(demo_config):
            # demo_config_spec = OrderedDict()
            demo_config_spec = {}
            for k in demo_config:
                if isinstance(demo_config[k], dict):
                    demo_config_spec[k] = to_demo_config_spec(demo_config[k])
                else:
                    v = demo_config[k]
                    if isinstance(v, str):
                        _type = "string"
                    elif isinstance(v, bool):
                        _type = "boolean"
                    elif isinstance(v, float) or isinstance(v, int):
                        _type = "number"
                    elif isinstance(v, list):
                        _type = "list"
                    else:
                        print("failing")
                        return abort(400)
                    demo_config_spec[k] = {"value": v, "type": _type}
            return demo_config_spec

        return json.dumps(to_demo_config_spec(demo_config))

    @app.post("/api/demo_config/set")
    def set_demo_config():
        data = request.get_json()
        if "demo" not in data or "config" not in data:
            return abort(400)
        selected_demo = data["demo"]
        demo_config = data["config"]
        try:
            OmegaConf.create(demo_config)
        except Exception:
            return abort(400)
        demo_runner.write_config(
            selected_demo, OmegaConf.to_yaml(OmegaConf.create(demo_config))
        )
        return ""

    socketio.init_app(app)
    socketio.run(app, host="0.0.0.0", port=9000)
