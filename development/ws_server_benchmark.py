import typing as t
import threading
from pathlib import Path
import queue
import time

from flask import Flask
from flask_socketio import SocketIO
from datetime import datetime

folder_path = Path(__file__).parent

app = Flask(
    __name__,
    template_folder=(folder_path / "templates").as_posix(),
    static_folder=(folder_path / "static").as_posix(),
)
socketio = SocketIO(async_mode="threading", cors_allowed_origins="*")

socketio_namespace = "/image_meta"
socketio_image_meta_data_queue = queue.Queue()

@socketio.on("connect", namespace=socketio_namespace)
def connect():
    pass

def socketio_emit_image_meta():
    while True:
        time.sleep(0.1)
        (frame, metadata) = ("small".encode("utf-8"), {"meta": "data"})
        # (frame, metadata) = socketio_image_meta_data_queue.get()
        frame = t.cast(bytes, frame)
        # print(str(datetime.now()))
        socketio.emit(
            "recv_image_meta",
            {"frame": frame.decode("utf-8"), "metadata": metadata, "time": str(datetime.now())},
            namespace=socketio_namespace,
        )

socketio_thread = threading.Thread(target=socketio_emit_image_meta, daemon=True)
socketio_thread.start()

socketio.init_app(app)
socketio.run(app, host="0.0.0.0", port="9000")
