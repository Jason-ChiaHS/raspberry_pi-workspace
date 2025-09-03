import statistics
import queue
import threading
import typing as t
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from sqlite3 import Cursor

import cv2
import numpy as np

from sdk.sqlite_adapter import connect_to_sqlite
from sdk.helpers.logger import logger

from .reid_models.arcface import ArcFace
from .reid_models.scrfd import SCRFD
from .reid_models.helpers import compute_similarity
from .utils import add_black_padding_to_image_opencv


class DataSampler:
    def __init__(self, min_samples: int = 5, max_samples: int = 5):
        self.min_samples = min_samples
        self.max_samples = max_samples

    def update_samples(self, samples: list, sample):
        samples.append(sample)
        if len(samples) > self.max_samples:
            samples.pop(0)
        return samples

    def get_median(self, samples: list):
        return statistics.median(samples)

    def get_max_count(self, samples: list):
        return max(samples, key=samples.count)


class CurrentTrack:
    def __init__(
        self,
        data_sampler: DataSampler,
        track_id: int,
        enter_time: datetime,
    ):
        self.missed_frames = 0
        self.data_sampler = data_sampler

        self.track_id = track_id
        self.enter_time = enter_time
        self.last_exit_time: t.Optional[datetime] = None
        self.ages: list[int] = []
        self.genders: list[str] = []

        self.best_gaze_metric


@dataclass
class DetectionWithTrackId:
    score: float
    bbox: np.array
    age: int
    gender: str
    yaw: float
    pitch: float
    is_frontal: bool
    # "score": score.item(),
    # "bbox": bbox,
    # "label": label,
    # "age": int(age.item()),
    # "gender": gender,
    # "yaw": headpose[0].item(),
    # "pitch": headpose[1].item(),
    # "is_frontal": abs(headpose[1].item())


@dataclass
class DetectionFrame:
    dt: datetime
    gaze: bool


@dataclass
class LastDetection:
    track_id: int
    enter_time: datetime
    exit_time: datetime
    longest_gaze: float
    face_img: np.array
    gender: str
    age: float
    gaze_metric: float

    score: float
    reid: int

    def to_dict(self) -> dict:
        # Everything except face_img
        return {
            "track_id": self.track_id,
            "enter_time": self.enter_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "longest_gaze": self.longest_gaze,
            "face_img": self.face_img,
            "gender": self.gender,
            "age": self.age,
            "gaze_metric": self.gaze_metric,
            "score": self.score,
            "reid": self.reid,
        }


class HistoricalDetectionsDBHandler:
    def __init__(self, config):
        self.db_path = Path(config.gaze.historical_detections.db_path)
        self.face_imgs_folder_path = Path(
            config.gaze.historical_detections.face_imgs_folder_path
        )
        self.face_imgs_folder_path.mkdir(parents=True, exist_ok=True)

        self.jobs = queue.Queue()
        self.thread = threading.Thread(target=self.handle_jobs)
        self.thread.start()

    def handle_jobs(self):
        with connect_to_sqlite(self.db_path) as con:
            cur = con.cursor()
            self.create_table(cur)
            con.commit()
            while (job := self.jobs.get()) is not None:
                data = job
                insert_statement = "INSERT INTO historical_detections VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)"
                cur.executemany(insert_statement, [data])
                con.commit()

    def create_table(self, cur: Cursor):
        cur.execute(
            "CREATE TABLE IF NOT EXISTS historical_detections(track_id INT NOT NULL, enter_time datetime NOT NULL, exit_time datetime NOT NULL, longest_gaze FLOAT NOT NULL, face_img_path TEXT NOT NULL, gender INT NOT NULL, age INT NOT NULL, gaze_metric FLOAT NOT NULL, score FLOAT NOT NULL)"
        )

    def queue_last_detections(self, last_detections: list[LastDetection]):
        # Thread the image saving
        with ThreadPoolExecutor(max_workers=3) as executor:
            for last_detection in last_detections:
                executor.submit(self.save_face_img_to_disk, last_detection)

    def save_face_img_to_disk(self, detection: LastDetection):
        img = detection.face_img
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        face_img_path = self.face_imgs_folder_path / f"{detection.track_id}.png"
        cv2.imwrite(face_img_path.as_posix(), img)
        self.jobs.put(
            (
                detection.track_id,
                detection.enter_time,
                detection.exit_time,
                detection.longest_gaze,
                face_img_path.as_posix(),
                detection.gender,
                detection.age,
                detection.gaze_metric,
                detection.score,
            )
        )


class HistoricalDetectionsProcessor:
    def __init__(self, config):
        self.detections_max_size = 500
        self.score_threshold_for_face_img = 0.95
        self.last_detections: list[LastDetection] = []
        self.seen_detections = {}

        self.db_handler = HistoricalDetectionsDBHandler(config)

        self.scrfd_model = SCRFD(
            "./demos/gaze-v2/models/det_500m.onnx", input_size=(640, 640)
        )
        self.arcface_model = ArcFace("./demos/gaze-v2/models/w600k_mbf.onnx")
        self.reids = []  # [idx: {embeddings: [{embedding, score}], average_embedding}]
        # the list of embedding and score is to get a weighted average of the embeddings, as a lower score usually indicates a poor quality of the face img
        self.reid_similarity_threshold = 0.4
        self.reid_score_threshold = 0.65

    def update_last_detections(self, track_ids, detections):
        dt = datetime.now()
        track_ids = set(track_ids)
        detections = {detection["track_id"]: detection for detection in detections}
        new_tracks = set(
            filter(lambda track_id: track_id not in self.seen_detections, track_ids)
        )
        new_tracks = set(filter(lambda track_id: track_id in detections, new_tracks))

        # Create new seen detections
        for track_id in new_tracks:
            detection = detections[track_id]
            self.seen_detections[track_id] = {
                "detection_frames": [DetectionFrame(dt, detection["is_frontal"])],
                **self.generate_seen_detection_metadata(detection),
            }

        # For all the remaining detections
        # If we have seen records of them, update them
        # If we also have a detection, check if we can update the metadata as well
        remaining_tracks = track_ids.difference(new_tracks)
        for track_id in remaining_tracks:
            # Ignore track_ids without any past or current detections
            if track_id in self.seen_detections:
                if track_id in detections:
                    detection = detections[track_id]
                    self.seen_detections[track_id]["detection_frames"].append(
                        DetectionFrame(dt, detection["is_frontal"])
                    )
                    # Update seen_detection_metadata if gaze_metric score is lower
                    seen_detection_metadata = self.generate_seen_detection_metadata(
                        detection
                    )
                    # The idea is that we want better score face_imgs until it passes the score_threshold_for_face_img
                    # Then we compare the gaze_metric metric
                    if (
                        self.seen_detections[track_id]["score"]
                        < self.score_threshold_for_face_img
                    ):
                        # Ignore gaze_metric metric here since a better score usually has a "better" face
                        if (
                            seen_detection_metadata["score"]
                            > self.seen_detections[track_id]["score"]
                        ):
                            self.seen_detections[track_id]["gaze_metric"] = (
                                seen_detection_metadata["gaze_metric"]
                            )
                            self.seen_detections[track_id]["score"] = (
                                seen_detection_metadata["score"]
                            )
                            self.seen_detections[track_id]["metadata"] = (
                                seen_detection_metadata["metadata"]
                            )
                    else:
                        if (
                            seen_detection_metadata["gaze_metric"]
                            < self.seen_detections[track_id]["gaze_metric"]
                        ):
                            self.seen_detections[track_id]["gaze_metric"] = (
                                seen_detection_metadata["gaze_metric"]
                            )
                            self.seen_detections[track_id]["score"] = (
                                seen_detection_metadata["score"]
                            )
                            self.seen_detections[track_id]["metadata"] = (
                                seen_detection_metadata["metadata"]
                            )

                        else:
                            # Skip since the detection model cannot pickup
                            logger.warning(
                                f"Unable to generate embedding vector for track: {track_id}"
                            )
                    # WIP
                    # First time we cross the threshold with no reid assigned
                    if False:
                        # if self.seen_detections[track_id]["score"] > self.reid_score_threshold and "reid" not in self.seen_detections[track_id]:
                        # If we beat the previous score, we generate an embedding vector for the face
                        # Get the image from self.seen_detections track_id (since it will have the best face)
                        face_img = self.seen_detections[track_id]["metadata"][
                            "face_img"
                        ]
                        cv2_face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                        # Add a perimeter of 50 black pixels around the img (So the detection can pick up and generate the kps)
                        padded_cv2_face_img = add_black_padding_to_image_opencv(
                            cv2_face_img, 50
                        )
                        _, kpss = self.scrfd_model.detect(
                            padded_cv2_face_img, max_num=1
                        )
                        if len(kpss) != 0:
                            embedding = self.arcface_model(padded_cv2_face_img, kpss[0])

                            # Find closest reid with close enough similiarity
                            max_similarity = 0
                            best_match_reid = -1
                            for reid, reid_metadata in enumerate(self.reids):
                                reid_embedding = reid_metadata["embedding"]
                                similarity = compute_similarity(
                                    reid_embedding, embedding
                                )
                                if (
                                    similarity > max_similarity
                                    and similarity > self.reid_similarity_threshold
                                ):
                                    max_similarity = similarity
                                    best_match_reid = reid
                            if best_match_reid != -1:
                                self.seen_detections[track_id]["reid"] = best_match_reid
                            else:
                                # No reid match, so we create a new reid
                                self.reids.append(
                                    {
                                        "embedding": embedding,
                                        "score": seen_detection_metadata["score"],
                                    }
                                )
                else:
                    self.seen_detections[track_id]["detection_frames"].append(
                        DetectionFrame(dt, False)
                    )

        # Check if we have any seen tracks that are not currently being tracked
        # Remove them and calculate last detection
        last_detections = []
        for removed_track_id in set(self.seen_detections.keys()).difference(track_ids):
            seen_detection = self.seen_detections[removed_track_id]
            # Get the longest gaze time
            is_gaze = False
            longest_gaze_so_far = 0
            gaze_starting_time = None
            for detection_frame in seen_detection["detection_frames"]:
                if is_gaze:
                    if not detection_frame.gaze:
                        is_gaze = False
                        gaze_time = detection_frame.dt - gaze_starting_time
                        gaze_time = gaze_time.total_seconds()
                        longest_gaze_so_far = max(longest_gaze_so_far, gaze_time)
                else:
                    if detection_frame.gaze:
                        is_gaze = True
                        gaze_starting_time = detection_frame.dt

            # Usually there are a few frames of no gaze as the buffer min count before a track is dropped
            if is_gaze:
                gaze_time = detection_frame.dt - gaze_starting_time
                gaze_time = gaze_time.total_seconds()
                longest_gaze_so_far = max(longest_gaze_so_far, gaze_time)
            last_detections.append(
                LastDetection(
                    removed_track_id,
                    seen_detection["detection_frames"][0].dt,
                    seen_detection["detection_frames"][-1].dt,
                    longest_gaze_so_far,
                    seen_detection["metadata"]["face_img"],
                    seen_detection["metadata"]["gender"],
                    seen_detection["metadata"]["age"],
                    seen_detection["gaze_metric"],
                    seen_detection["score"],
                    seen_detection["reid"] if "reid" in seen_detection else -1,
                )
            )
            del self.seen_detections[removed_track_id]
        self.db_handler.queue_last_detections(last_detections)
        self.last_detections.extend(last_detections)

        if len(self.last_detections) > 50:
            self.last_detections = self.last_detections[
                len(self.last_detections) - 50 :
            ]

    def generate_seen_detection_metadata(self, detection):
        pitch = detection["pitch"]
        yaw = detection["yaw"]
        return {
            "metadata": {
                # We take the age and gender from the sort algo if possible
                "age": detection["age"]
                if detection["final_age"] is None
                else detection["final_age"],
                "gender": detection["gender"]
                if detection["final_gender"] is None
                else detection["final_gender"],
                "face_img": detection["face_img"],
                "is_frontal": detection["is_frontal"],
                "pitch": pitch,
                "yaw": yaw,
                "score": detection["score"],
            },
            # Calculated, smaller is better, penalise as follows
            # NOTE: You can tune the weightage here
            # 1. large pitch and yaw
            # 2. Asymmetric large and small
            # 3. small pitch and yaw
            "gaze_metric": max(abs(pitch), abs(yaw)) + (abs(pitch) + abs(yaw)),
            "score": detection["score"],
        }


@dataclass
class GazePeopleData:
    dt_time: datetime
    py_time: float
    gaze_people: int
    non_gaze_people: int

    def to_dict(self) -> dict:
        return {
            "dt_time": self.dt_time.isoformat(),
            "py_time": self.py_time,
            "gaze_people": self.gaze_people,
            "non_gaze_people": self.non_gaze_people,
        }


class GazePeopleProcessor:
    def __init__(self):
        self.gaze_seconds_interval = 1
        self.gaze_people_datas: list[GazePeopleData] = []
        self.gaze_people_data_ring_size = 20
        self.gaze_people_data_buffer: list[GazePeopleData] = []
        self.latest_gaze_people_data: t.Optional[GazePeopleData] = None

    def update_gaze_people_datas(self, detections, frame_time):
        dt_time = datetime.now()
        gaze_people = 0
        for detection in detections:
            if detection["is_frontal"]:
                gaze_people += 1
        non_gaze_people = len(detections) - gaze_people
        gaze_people_data = GazePeopleData(
            dt_time,
            frame_time,
            gaze_people,
            non_gaze_people,
        )
        self.latest_gaze_people_data = gaze_people_data
        if len(self.gaze_people_data_buffer) == 0:
            # Nothing in buffer, so we start
            self.gaze_people_data_buffer.append(gaze_people_data)
            return

        # Check against the first in the buffer list, get the diff
        earliest_gaze_people_data = self.gaze_people_data_buffer[0]
        earliest_py_time = earliest_gaze_people_data.py_time
        if frame_time - earliest_py_time > self.gaze_seconds_interval:
            # Calculate average from all in buffer, add to datas
            total_gaze_people = 0
            total_non_gaze_people = 0
            for gpd in self.gaze_people_data_buffer:
                total_gaze_people += gpd.gaze_people
                total_non_gaze_people += gpd.non_gaze_people
            earliest_gaze_people_data.gaze_people = round(
                total_gaze_people / len(self.gaze_people_data_buffer)
            )
            earliest_gaze_people_data.non_gaze_people = round(
                total_non_gaze_people / len(self.gaze_people_data_buffer)
            )
            self.append_gaze_people_datas(earliest_gaze_people_data)

            # Past interval, start a new buffer
            self.gaze_people_data_buffer = [gaze_people_data]
        else:
            self.gaze_people_data_buffer.append(gaze_people_data)

    def append_gaze_people_datas(self, gaze_people_data: GazePeopleData):
        if (
            len(self.gaze_people_datas) > 0
            and self.gaze_people_datas[-1].dt_time == gaze_people_data.dt_time
        ):
            # If we get duplicate dt_time because the frames are too slow
            # Don't change anything because it causes duplicate bars to show up on the front-end
            return
        else:
            self.gaze_people_datas.append(gaze_people_data)
        if len(self.gaze_people_datas) > self.gaze_people_data_ring_size:
            del self.gaze_people_datas[0]

    def gaze_people_datas_to_dict(self):
        return [gpd.to_dict() for gpd in self.gaze_people_datas]
