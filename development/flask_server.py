from queue import Queue
from flask import Flask, request, Response, Request
import json
from gevent.pywsgi import WSGIServer
import traceback
import threading


class ImageMetaFlaskServer:
    def __init__(
        self,
        name: str,
        data_queue: Queue,
    ):
        self.data_queue = data_queue
        self.app = Flask(name)
        self.lock = threading.Lock()
        self.buffering_data = {}

        @self.app.route("/trigger/image/<filename>", methods=["PUT"])
        def update_image(filename):
            try:
                # print("WTA Image Rcv")
                img = self.process_image_request(request)
                with self.lock:
                    if self.buffering_data.get(filename) is not None:
                        self.data_queue.put(
                            {
                                "frame": img,
                                "metadata": self.buffering_data[filename],
                                "filename": filename,
                            }
                        )
                        del self.buffering_data[filename]
                    else:
                        self.buffering_data[filename] = img
                return Response(response=json.dumps({"status": "success"}), status=200)
            except Exception:
                traceback.print_exc()

        @self.app.route("/trigger/meta/<filename>", methods=["PUT"])
        def update_meta(filename):
            try:
                # print("WTA Meta Rcv")
                metadata = self.process_metadata_request(request)
                with self.lock:
                    if self.buffering_data.get(filename) is not None:
                        self.data_queue.put(
                            {
                                "frame": self.buffering_data[filename],
                                "metadata": metadata,
                                "filename": filename,
                            }
                        )
                        del self.buffering_data[filename]
                    else:
                        self.buffering_data[filename] = metadata
                return Response(response=json.dumps({"status": "success"}), status=200)
            except Exception:
                traceback.print_exc()

    def process_image_request(self, req: Request):
        """
        Should overide depending on how image data is sent from the SDK
        I.e Deserialize the image data from the request
        """
        return req.get_data()

    def process_metadata_request(self, req: Request):
        """
        Should overide depending on how metadata needs to be processed from the SDK
        I.e Deserialize the metadata from the request
        """
        return req.get_json()

    def start_server(self):
        data = {"HTTP_LISTENER_PORT": 8010}
        if "HTTP_LISTENER_PORT" not in data:
            raise Exception()
        self.t = FlaskThread(self.app, "0.0.0.0", data["HTTP_LISTENER_PORT"])
        self.t.start()

    def stop_server(self):
        self.t.stop()

class FlaskThread(threading.Thread):
    def __init__(self, app, host="0.0.0.0", port=8010):
        super().__init__()
        self.host = host
        self.port = port
        self.server = None
        self.running = threading.Event()
        self.app = app

    def run(self):
        """Start the Flask server."""
        print(f"Listening on {self.host}:{self.port}")
        self.server = WSGIServer((self.host, self.port), self.app)
        self.running.set()  # Signal that the server is running
        self.server.serve_forever()

    def stop(self):
        """Stop the Flask server."""
        if self.running:
            self.server.stop()
            self.running.clear()  # Signal that the server has stopped

if __name__ == "__main__":
    data_queue = Queue()
    image_meta_flask_server = ImageMetaFlaskServer("test", data_queue=data_queue)
    image_meta_flask_server.start_server()
    while data := data_queue.get():
        print(data["metadata"], data["filename"])
