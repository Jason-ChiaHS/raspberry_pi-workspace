import base64
import queue
import threading
import time
import typing as t
from datetime import datetime
from pathlib import Path
from concurrent import futures

import cv2
import requests

from sdk.helpers.artifact_generator import CustomProfiler
from sdk.helpers.logger import logger
from sdk.sqlite_adapter import connect_to_sqlite

from ..common import AdditionalDemoConfig, FaceBodyTrack, HistorialDetectionEntry
from ..config import DB
from ..models.face_reid.arcface import ArcFace
from ..models.mobilenet_age import MobileNetAge
from ..models.mobilenet_face_landmark import MobileNetV2FaceLandmark
from ..sqlite_db import SDKArtifactDir
from .face_reid import (
    FaceReIDBaseDatabase,
    FaceReIDHTTPCentralServerDatabase,
    FaceReIDMemoryDatabase,
    FaceReIDSQLiteDatabase,
)
from .metadata import (
    AgeMetadataProcessor,
    FaceImgMetadataProcessor,
    FaceReIDMetadataProcessor,
    GazeTimeMetadataProcessor,
    GenderMetadataProcessor,
)


class ProcessedTrack:
    """
    We take in all possible models to then pass them to the respective metadataprocessor
    """

    def __init__(
        self,
        config: AdditionalDemoConfig,
        track_id: int,
        entry: datetime,
        age_enhancement_model: MobileNetAge,
        face_embedding_model: ArcFace,
        face_landmark_model: MobileNetV2FaceLandmark,
        face_reid_database: FaceReIDBaseDatabase,
    ):
        self.config = config
        self.entry = entry
        self.track_id = track_id

        self.age = AgeMetadataProcessor(config, age_enhancement_model)
        self.gender = GenderMetadataProcessor(config)
        self.gaze_time = GazeTimeMetadataProcessor(config)
        self.face_img = FaceImgMetadataProcessor(config)
        self.face_reid = FaceReIDMetadataProcessor(
            config, face_embedding_model, face_landmark_model, face_reid_database
        )

    def init_with_face_body_track(
        config: AdditionalDemoConfig,
        track_id: int,
        entry_dt: datetime,
        face_body_track: FaceBodyTrack,
        img,
        age_enhancement_model: MobileNetAge,
        face_embedding_model: ArcFace,
        face_landmark_model: MobileNetV2FaceLandmark,
        face_reid_database: FaceReIDBaseDatabase,
    ):
        processed_track = ProcessedTrack(
            config,
            track_id,
            entry_dt,
            age_enhancement_model,
            face_embedding_model,
            face_landmark_model,
            face_reid_database,
        )
        processed_track.update(face_body_track, img, entry_dt)
        return processed_track

    def update(self, face_body_track: FaceBodyTrack, img, dt: datetime):
        """
        img: RGB cv2 img
        dt: synced datetime from the TracksProcessor
        """
        self.age.update(face_body_track.face, img)
        self.gender.update(face_body_track.face)
        self.gaze_time.update(face_body_track.face, dt)
        self.face_img.update(face_body_track.face, img)
        self.face_reid.update(face_body_track.face, img)

    def update_without_face_body_track(self, dt: datetime):
        """
        dt: synced datetime from the TracksProcessor
        """
        self.gaze_time.update_without_face_detection(dt)


class HistoricalProcessedTrack:
    """
    The reason for this being a different class is to free up the mem for the ProcessedTrack
    Since it has a lot of references and temp data
    """

    def __init__(self, processed_track: ProcessedTrack, exit: datetime):
        self.entry = processed_track.entry
        self.exit = exit
        self.track_id = processed_track.track_id

        self.age = processed_track.age.get()
        self.gender = processed_track.gender.get()
        self.gaze_time = processed_track.gaze_time.get(exit)
        self.face_img = processed_track.face_img.get()
        self.face_reid = processed_track.face_reid.get()

    def is_any_missing(self):
        """
        Returns True if the track contains any missing fields
        """
        return (
            self.age is None
            or self.gender is None
            or self.gaze_time is None
            or self.face_img is None
            or self.face_reid is None
        )


class HistoricalProcessedTrackDatabase:
    def save_entry(self, entry: HistoricalProcessedTrack):
        pass


class HistoricalProcessedTrackSQLite(SDKArtifactDir, HistoricalProcessedTrackDatabase):
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
                    face_reid INT
                )
                """)
                con.commit()

        self.jobs = queue.Queue()
        self.thread = threading.Thread(target=self.handle_jobs, daemon=True)
        self.thread.start()

    def handle_jobs(self):
        with connect_to_sqlite(self.db_file) as con:
            cur = con.cursor()
            while (job := self.jobs.get()) is not None:
                entry = t.cast(HistoricalProcessedTrack, job)
                face_img_file_path: t.Optional[str] = None
                if entry.face_img is not None:
                    img = cv2.cvtColor(entry.face_img, cv2.COLOR_RGB2BGR)
                    face_img_path = self.face_imgs_folder_path / f"{entry.track_id}.png"
                    face_img_file_path = face_img_path.as_posix()
                    cv2.imwrite(face_img_file_path, img)

                cur = con.cursor()
                insert_statement = (
                    "INSERT INTO historical_detections VALUES(?, ?, ?, ?, ?, ?, ?, ?)"
                )
                cur.execute(
                    insert_statement,
                    [
                        entry.track_id,
                        entry.entry,
                        entry.exit,
                        entry.gaze_time,
                        face_img_file_path,
                        entry.gender,
                        entry.age,
                        entry.face_reid,
                    ],
                )
                con.commit()

    def save_entry(self, entry: HistoricalProcessedTrack):
        self.jobs.put(entry)


class HistoricalProcessedTrackHTTPCentral(HistoricalProcessedTrackDatabase):
    """
    Since the saving of the entries does not need to be tied to the performance of the demo
    We can offload it to threads and allow the main thread to continue
    Allows for better real time performance
    """

    def __init__(self, config: AdditionalDemoConfig):
        self.base_url = config.gaze.tracker.http_central_client.base_url
        self.cam_name = config.gaze.tracker.cam_name

        self.max_workers = 3
        self.executor = futures.ThreadPoolExecutor(max_workers=self.max_workers)

        self.profiling = config.profiling

    def send_to_http_central_server(self, entry: HistoricalProcessedTrack):
        img = entry.face_img
        if img is not None:
            _, encoded_image = cv2.imencode(".jpg", img)
            frame = base64.b64encode(encoded_image.tobytes()).decode("utf-8")
        else:
            frame = img  # None
        data = HistorialDetectionEntry(
            track_id=entry.track_id,
            entry=entry.entry.isoformat(),
            exit=entry.exit.isoformat(),
            gaze_time=entry.gaze_time,
            gender=entry.gender,
            age=entry.age,
            face_reid=entry.face_reid,
            img=frame,
            cam_name=self.cam_name
        )
        try:
            if self.profiling:
                start = time.perf_counter()
            res = requests.post(
                f"{self.base_url}/historical_detections",
                json=data.model_dump(),
                timeout=2,
            )
            if self.profiling:
                end = time.perf_counter()
                logger.info(f"http time: {end - start}")
            return res.json()
        except:  # noqa: E722
            logger.warning("Failed to connect to FaceReIDHTTPCentralServerDatabase")
            pass

    def save_entry(self, entry):
        self.executor.submit(self.send_to_http_central_server, entry)


class TracksProcessor:
    def __init__(self, config: AdditionalDemoConfig):
        self.config = config
        self.lock = threading.Lock()

        # trackid: ProcessedTrack
        self.current_tracks: dict[int, ProcessedTrack] = {}
        self.past_tracks: list[HistoricalProcessedTrack] = []

        # Declaring the models here to be passed as reference to the process_tracks MetadataProcessors
        # This is to cache and reuse the models since they are expensive to initalize
        # We initalize them regardless of if we are using them
        # We are ok with this, since we expect the demo to run with all models running at the same time without any issue
        self.age_enhancement_model = MobileNetAge(
            self.config.gaze.tracker.age.enhancement.model_path
        )
        self.face_embedding_model = ArcFace(
            self.config.gaze.tracker.face_reid.face_embedding_model_path
        )
        self.face_landmark_model = MobileNetV2FaceLandmark(
            self.config.gaze.tracker.face_reid.face_landmark_model_path
        )

        # Face ReID
        if self.config.gaze.tracker.face_reid.db == DB.SQLITE.value:
            self.face_reid_database = FaceReIDSQLiteDatabase(config)
        elif self.config.gaze.tracker.face_reid.db == DB.HTTP_CENTRAL_SERVER.value:
            self.face_reid_database = FaceReIDHTTPCentralServerDatabase(config)
        else:
            # Default to memory
            self.face_reid_database = FaceReIDMemoryDatabase(config)

        if self.config.gaze.tracker.db == DB.SQLITE.value:
            self.historical_processed_track_sqlite = HistoricalProcessedTrackSQLite(
                config
            )
        elif self.config.gaze.tracker.db == DB.HTTP_CENTRAL_SERVER.value:
            self.historical_processed_track_sqlite = (
                HistoricalProcessedTrackHTTPCentral(config)
            )

    def per_frame_update(
        self, track_ids: list[int], face_body_tracks: list[FaceBodyTrack], img
    ):
        """
        Should be called every inference frame
        Updates the tracks

        track_ids: list of all track ids that are still tracked
                   We let the tracker handle when to drop the track, so when the track_id does not appear here
                   We assume the track has been dropped
        face_body_tracks: list of all FaceBodyTrack
        """
        with self.lock:
            dt = datetime.now()
            track_ids = set(track_ids)
            face_body_tracks: dict[int, FaceBodyTrack] = {
                face_body_track.track_id: face_body_track
                for face_body_track in face_body_tracks
            }

            # remove untracked tracks
            untracked_ids = set(self.current_tracks.keys()).difference(track_ids)
            for untracked_id in untracked_ids:
                self.remove_track(untracked_id, dt)

            # current track_ids that we have updated face_body_tracks
            current_tracks_with_update = set(self.current_tracks.keys()).intersection(
                set(face_body_tracks.keys())
            )
            for track_id in current_tracks_with_update:
                self.current_tracks[track_id].update(
                    face_body_tracks[track_id], img, dt
                )

            # Currently tracked, but no new face_body_track
            current_tracks_without_update = set(self.current_tracks.keys()).difference(
                set(face_body_tracks.keys())
            )
            for track_id in current_tracks_without_update:
                self.current_tracks[track_id].update_without_face_body_track(dt)

            # For new tracks, add them
            new_track_ids = track_ids.difference(set(self.current_tracks.keys()))
            for new_track_id in new_track_ids:
                if new_track_id in face_body_tracks:
                    self.current_tracks[new_track_id] = (
                        ProcessedTrack.init_with_face_body_track(
                            self.config,
                            new_track_id,
                            dt,
                            face_body_tracks[new_track_id],
                            img,
                            self.age_enhancement_model,
                            self.face_embedding_model,
                            self.face_landmark_model,
                            self.face_reid_database,
                        )
                    )
                else:
                    self.current_tracks[new_track_id] = ProcessedTrack(
                        self.config,
                        new_track_id,
                        dt,
                        self.age_enhancement_model,
                        self.face_embedding_model,
                        self.face_landmark_model,
                        self.face_reid_database,
                    )

    def remove_track(self, track_id, dt):
        # Remove untracked track_id
        historical_processed_track = HistoricalProcessedTrack(
            self.current_tracks[track_id], dt
        )
        del self.current_tracks[track_id]

        if (
            self.config.gaze.tracker.discard_any_empty_historical_tracks
            and historical_processed_track.is_any_missing()
        ):
            return

        if len(self.past_tracks) >= self.config.gaze.tracker.max_historical_buffer:
            self.past_tracks.pop(0)
        self.past_tracks.append(historical_processed_track)

        # Additional saving happens here
        if self.config.gaze.tracker.db != DB.MEMORY.value:
            self.historical_processed_track_sqlite.save_entry(
                historical_processed_track
            )
