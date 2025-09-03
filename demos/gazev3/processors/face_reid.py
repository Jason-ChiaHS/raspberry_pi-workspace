import base64
import typing as t
import cv2
import uuid
from dataclasses import dataclass
import queue
import threading
from pathlib import Path
import sqlite_vec
import time

import numpy as np
import requests

from ..common import AdditionalDemoConfig, FaceReIDEntry
from ..sqlite_db import SDKArtifactDir
from sdk.sqlite_adapter import connect_to_sqlite
from sdk.helpers.logger import logger


@dataclass
class FaceReID:
    embedding: np.ndarray
    face: np.array


class FaceReIDBaseDatabase:
    def __init__(self, similarity_threshold: float):
        self.similarity_threshold = similarity_threshold

    def face_reid_entry(self, embedding: np.ndarray, img: np.ndarray) -> t.Optional[list[int]]:
        """
        Given the face embedding and img
        Returns the list of track_ids that match above the threshold

        embedding: [1, 512]
        """
        pass


class FaceReIDHTTPCentralServerDatabase(FaceReIDBaseDatabase):
    def __init__(self, config: AdditionalDemoConfig):
        super().__init__(config.gaze.tracker.face_reid.similarity_threshold)
        self.base_url = config.gaze.tracker.http_central_client.base_url

        self.profiling = config.profiling

    def face_reid_entry(self, embedding, img):
        """
        Has to be done synchronously because we need the face_reid back
        """
        _, encoded_image = cv2.imencode(".jpg", img)
        frame = base64.b64encode(encoded_image.tobytes()).decode("utf-8")
        data = FaceReIDEntry(embedding=embedding[0].tolist(), img=frame)
        try:
            if self.profiling:
                start = time.perf_counter()
            res = requests.post(
                f"{self.base_url}/face_reid",
                json=data.model_dump(),
                timeout=2
            )
            if self.profiling:
                end = time.perf_counter()
                logger.info(f"http time: {end - start}")
            return res.json()
        except:  # noqa: E722
            logger.warning("Failed to connect to FaceReIDHTTPCentralServerDatabase")
            return None


class FaceReIDMemoryDatabase(FaceReIDBaseDatabase):
    def __init__(self, config: AdditionalDemoConfig):
        super().__init__(config.gaze.tracker.face_reid.similarity_threshold)
        self.face_reids: list[FaceReID] = []  # idx: face_reid, [FaceReID]

    def face_reid_entry(self, embedding: np.ndarray, img: np.ndarray) -> list[int]:
        if len(self.face_reids) == 0:
            # empty just add embedding
            self.face_reids.append(FaceReID(embedding, img.copy()))
            return [0]
        similarity = self.cosine_similarity_one_to_many(
            embedding, [e.embedding.flatten() for e in self.face_reids]
        )
        above_threshold = similarity > self.similarity_threshold
        face_reids = np.where(above_threshold)[0].tolist()
        if len(face_reids) > 0:
            return face_reids
        else:
            self.face_reids.append(FaceReID(embedding, img.copy()))
            return [len(self.face_reids) - 1]

    def cosine_similarity_one_to_many(
        self, target_embedding: np.ndarray, embeddings: list[np.ndarray]
    ):
        """
        target_embedding: should have shape (1, X)
        embeddings: should be a list with each embedding have shape (X,)
        """
        np_vector_a = target_embedding.flatten()
        np_list_of_vectors_b = np.array(embeddings)

        if np_list_of_vectors_b.shape[1] != np_vector_a.shape[0]:
            raise ValueError("All vectors must have the same dimensions.")

        norm_a = np.linalg.norm(np_vector_a)
        norms_b = np.linalg.norm(np_list_of_vectors_b, axis=1)

        dot_products = np.dot(np_list_of_vectors_b, np_vector_a)

        denominator = norm_a * norms_b
        cosine_similarities = np.where(
            denominator == 0, 0.0, dot_products / denominator
        )

        return cosine_similarities


class FaceReIDSQLiteDatabase(FaceReIDBaseDatabase, SDKArtifactDir):
    def __init__(self, config: AdditionalDemoConfig):
        FaceReIDBaseDatabase.__init__(
            self, config.gaze.tracker.face_reid.similarity_threshold
        )
        SDKArtifactDir.__init__(self, config)

        self.k = 20  # Limit to how many embeddings to return

        self.db_file = self.artifact_folder / "face_reid.db"
        self.face_imgs_folder_path = self.artifact_folder / "face_reid_face_imgs"
        self.face_imgs_folder_path.mkdir(parents=True)

        if not Path(self.db_file).exists():
            with self.__connect_sqlite_vec(self.db_file) as con:
                cur = con.cursor()
                cur.execute("""
                CREATE VIRTUAL TABLE face_reid
                USING vec0(
                    id INTEGER PRIMARY KEY, path TEXT, embedding float[512] distance_metric=cosine
                )
                """)

                con.commit()

        self.jobs = queue.Queue()
        self.thread = threading.Thread(target=self.handle_jobs, daemon=True)
        self.thread.start()

    def __connect_sqlite_vec(self, db_file: Path):
        con = connect_to_sqlite(db_file)
        con.enable_load_extension(True)
        sqlite_vec.load(con)
        con.enable_load_extension(False)
        return con

    def handle_jobs(self):
        # Just to save the file down
        while (job := self.jobs.get()) is not None:
            img, img_path = job
            img = t.cast(np.ndarray, img)
            img_path = t.cast(Path, img_path)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_path.as_posix(), img)

    def face_reid_entry(self, embedding, img):
        with self.__connect_sqlite_vec(self.db_file) as con:
            cur = con.cursor()
            rows = cur.execute(
                """
                SELECT
                id, path, distance
                FROM face_reid
                WHERE embedding MATCH ?
                AND distance < ? AND k = ?
                ORDER BY distance
                """,
                [embedding, 1 - self.similarity_threshold, 20],
            ).fetchall()
            if len(rows) == 0:
                # empty insert and move on
                face_img_path = self.face_imgs_folder_path / f"{uuid.uuid4()}.png"
                cur.execute(
                    "INSERT INTO face_reid(path, embedding) VALUES (?, ?)",
                    [face_img_path.as_posix(), embedding],
                )
                id = cur.lastrowid
                con.commit()
                self.jobs.put((img, face_img_path))
                return [id]
            else:
                # return the list
                ids = [row[0] for row in rows]
                return ids
