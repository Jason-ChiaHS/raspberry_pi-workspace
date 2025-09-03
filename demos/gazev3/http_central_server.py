"""
FastAPI central server for all gazev3 demo related options
"""

import base64
import typing as t
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI

from sdk.sqlite_adapter import connect_to_sqlite

from .common import AdditionalDemoConfig, FaceReIDEntry, HistorialDetectionEntry
from .processors.face_reid import FaceReIDSQLiteDatabase
from .sqlite_db import SDKArtifactDir


class HistoricalDetectionSQLite(SDKArtifactDir):
    """
    The caller is responsible for ensuring if this is required as the initialization would create the folder and db
    """

    def __init__(self, config: AdditionalDemoConfig):
        super().__init__(config)
        self.db_file = self.artifact_folder / "tracks.db"
        self.face_imgs_folder_path = self.artifact_folder / "tracks_face_imgs"
        self.face_imgs_folder_path.mkdir(parents=True)
        if not Path(self.db_file).exists():
            with connect_to_sqlite(self.db_file) as con:
                cur = con.cursor()
                cur.execute("""
                CREATE TABLE IF NOT EXISTS historical_detections(
                    track_id INT NOT NULL,
                    entry datetime NOT NULL,
                    exit datetime NOT NULL,
                    gaze_time FLOAT,
                    face_img_path TEXT,
                    gender INT,
                    age INT,
                    face_reid INT,
                    cam_name TEXT NOT NULL
                )
                """)
                con.commit()


    def save_entry(self, entry: HistorialDetectionEntry):
        face_img_file_path: t.Optional[str] = None
        if entry.img is not None:
            img = base64.b64decode(entry.img)
            img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cam_face_img_folder = self.face_imgs_folder_path / entry.cam_name
            cam_face_img_folder.mkdir(parents=True, exist_ok=True)
            face_img_path = cam_face_img_folder / f"{entry.track_id}.png"

            face_img_file_path = face_img_path.as_posix()
            cv2.imwrite(face_img_file_path, img)

        with connect_to_sqlite(self.db_file) as con:
            cur = con.cursor()
            insert_statement = (
                "INSERT INTO historical_detections VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)"
            )

            cur.execute(
                insert_statement,
                [
                    entry.track_id,
                    datetime.fromisoformat(entry.entry),
                    datetime.fromisoformat(entry.exit),
                    entry.gaze_time,
                    face_img_file_path,
                    entry.gender,
                    entry.age,
                    entry.face_reid,
                    entry.cam_name
                ],
            )
            con.commit()


class HTTPCentralServer:
    def __init__(self, config: AdditionalDemoConfig):
        self.config = config
        self.face_reid_sqlite_db = FaceReIDSQLiteDatabase(self.config)
        self.historical_detection_sqlite = HistoricalDetectionSQLite(self.config)
        self.app = FastAPI()

        @self.app.post("/face_reid")
        async def post_face_reid(face_reid: FaceReIDEntry):
            img = base64.b64decode(face_reid.img)
            img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
            embedding = np.array([face_reid.embedding], dtype=np.float32)

            return self.face_reid_sqlite_db.face_reid_entry(embedding, img)

        @self.app.post("/historical_detections")
        async def post_historical_detections(
            historical_detection: HistorialDetectionEntry,
        ):
            self.historical_detection_sqlite.save_entry(historical_detection)
            return

        uvicorn.run(
            self.app,
            host="0.0.0.0",
            port=self.config.gaze.http_central_server.host_port,
        )
