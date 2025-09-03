"""
For FaceDetection MetadataProcessors
"""

import statistics
import typing as t
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from sdk.helpers.artifact_generator import CustomProfiler
from sdk.helpers.logger import logger

from ..common import AdditionalDemoConfig, FaceDetection
from ..models.face_reid.arcface import ArcFace
from ..models.face_reid.helpers import norm_crop_image
from ..models.mobilenet_age import MobileNetAge
from ..models.mobilenet_face_landmark import MobileNetV2FaceLandmark
from .face_reid import FaceReIDBaseDatabase


def get_face_img(
    bbox: np.ndarray, base_img: np.ndarray, face_expand_ratio: float
) -> t.Optional[np.ndarray]:
    """
    bbox: [x1, y1, x2, y2]
    base_img: cv2 whole img with the face in the bbox

    returns: None if the face_img has no shape, else a copy of the img
    """
    (x1, y1, x2, y2) = bbox
    x = x2 - x1
    y = y2 - y1
    fy1 = max(int(y1 - face_expand_ratio * y), 0)
    fy2 = min(
        int(y2 + face_expand_ratio * y),
        len(base_img[0]) - 1,
    )
    fx1 = max(int(x1 - face_expand_ratio * x), 0)
    fx2 = min(
        int(x2 + face_expand_ratio * x),
        len(base_img[1] - 1),
    )
    face_img = base_img[
        fy1:fy2,
        fx1:fx2,
    ][:, :, :3]

    empty_face = False
    for dim in face_img.shape:
        if dim == 0:
            empty_face = True

    if empty_face:
        return None
    else:
        return face_img.copy()


@dataclass
class MetadataMetric:
    """
    Metrics used to compare metadata "accuracy"
    """

    score: float
    pitch: float
    yaw: float

    def from_face_detection(face_detection: FaceDetection):
        return MetadataMetric(
            face_detection.score, face_detection.pitch, face_detection.yaw
        )


class FaceDetectionMetadataProcessor:
    def __init__(self, config: AdditionalDemoConfig):
        self.config = config

    def update(self, face_body_track: FaceDetection):
        pass

    def sort_metric(
        self, metric_metadata: list[tuple[MetadataMetric, any]]
    ) -> list[tuple[MetadataMetric, int]]:
        return sorted(metric_metadata, key=lambda m: m[0].score)


class AgeMetadataProcessor(FaceDetectionMetadataProcessor):
    def __init__(self, config, age_enhancement_model: MobileNetAge):
        super().__init__(config)
        self.above_score_threshold = False

        self.ages: list[tuple[MetadataMetric, int]] = []
        self.age_enhancement_config = self.config.gaze.tracker.age.enhancement
        self.min_samples = self.config.gaze.tracker.age.min
        self.max_samples = self.config.gaze.tracker.age.max
        self.accurate_face_threshold_config = (
            self.config.gaze.tracker.accurate_face_threshold
        )

        self.mobilenet_age = age_enhancement_model

    def update(self, face: FaceDetection, img):
        """
        img: cv2 RGB
        """
        # if not we only replace with higher score
        # if all are above the score threshold we do not change anymore
        if self.above_score_threshold:
            return

        # if we still have space, we will take if there are withing the gaze threshold
        if len(self.ages) < self.max_samples:
            if (
                self.accurate_face_threshold_config.yaw.min
                <= face.pitch
                <= self.accurate_face_threshold_config.yaw.max
                and self.accurate_face_threshold_config.pitch.min
                <= face.pitch
                <= self.accurate_face_threshold_config.pitch.max
            ):
                ages = self.ages
                ages.append(
                    (
                        MetadataMetric.from_face_detection(face),
                        self.calculate_age(face, img),
                    )
                )
                self.ages = self.sort_metric(ages)
                return
        # We are at max, we only want to replace with higher scores
        # By default the ages are sorted by scores
        if len(self.ages) == self.max_samples:
            if face.score > self.ages[0][0].score:
                ages = self.ages
                ages[0] = (
                    MetadataMetric.from_face_detection(face),
                    self.calculate_age(face, img),
                )
                self.ages = self.sort_metric(ages)
                # Check if we are done with stable ages
                if all(
                    map(
                        lambda age: age[0].score
                        >= self.accurate_face_threshold_config.score,
                        self.ages,
                    )
                ):
                    self.above_score_threshold = True

    def get(self) -> t.Optional[int]:
        if len(self.ages) < self.min_samples:
            return None

        # return median for now
        return statistics.median(map(lambda age: age[1], self.ages))

    def calculate_age(self, face: FaceDetection, img) -> int:
        """
        helper function to take into account mobilenet age enhancement model
        """
        if not self.age_enhancement_config.enable:
            return face.age

        face_img = get_face_img(
            face.bbox, img, self.age_enhancement_config.face_expand_ratio
        )
        if face_img is None:
            # if it somehow becomes an empty face, we return the original model result
            # Mainly to catch edge cases
            return face.age

        mobilenet_age = self.mobilenet_age.run_inference(face_img)

        return int(mobilenet_age.item())


class GenderMetadataProcessor(FaceDetectionMetadataProcessor):
    def __init__(self, config):
        super().__init__(config)
        self.above_score_threshold = False

        self.genders: list[tuple[MetadataMetric, str]] = []
        self.min_samples = self.config.gaze.tracker.gender.min
        self.max_samples = self.config.gaze.tracker.gender.max
        self.accurate_face_threshold_config = (
            self.config.gaze.tracker.accurate_face_threshold
        )

    def update(self, face: FaceDetection):
        # if not we only replace with higher score
        # if all are above the score threshold we do not change anymore
        if self.above_score_threshold:
            return

        # if we still have space, we will take if there are withing the gaze threshold
        if len(self.genders) < self.max_samples:
            if (
                self.accurate_face_threshold_config.yaw.min
                <= face.pitch
                <= self.accurate_face_threshold_config.yaw.max
                and self.accurate_face_threshold_config.pitch.min
                <= face.pitch
                <= self.accurate_face_threshold_config.pitch.max
            ):
                genders = self.genders
                genders.append((MetadataMetric.from_face_detection(face), face.gender))
                self.genders = self.sort_metric(genders)
                return
        # We are at max, we only want to replace with higher scores
        # By default the genders are sorted by scores
        if len(self.genders) == self.max_samples:
            if face.score > self.genders[0][0].score:
                genders = self.genders
                genders[0] = (MetadataMetric.from_face_detection(face), face.gender)
                self.genders = self.sort_metric(genders)
                # Check if we are done with stable genders
                if all(
                    map(
                        lambda gender: gender[0].score
                        >= self.accurate_face_threshold_config.score,
                        self.genders,
                    )
                ):
                    self.above_score_threshold = True

    def get(self) -> t.Optional[int]:
        if len(self.genders) < self.min_samples:
            return None

        # return median for now
        return statistics.mode(map(lambda gender: gender[1], self.genders))


class GazeTimeMetadataProcessor(FaceDetectionMetadataProcessor):
    def __init__(self, config):
        super().__init__(config)
        self.current_gaze_start: t.Optional[datetime] = None
        self.longest_gaze_so_far: t.Optional[float] = None
        self.missed_so_far = 0  # Tracks the number of frames with no gaze so far

        self.accurate_face_threshold_config = (
            self.config.gaze.tracker.accurate_face_threshold
        )
        self.lost_buffer = self.config.gaze.tracker.gaze.lost_buffer

    def update(self, face: FaceDetection, dt: datetime):
        """
        dt: datetime synced from tracks
        """
        if (
            self.accurate_face_threshold_config.yaw.min
            <= face.pitch
            <= self.accurate_face_threshold_config.yaw.max
            and self.accurate_face_threshold_config.pitch.min
            <= face.pitch
            <= self.accurate_face_threshold_config.pitch.max
        ):
            if self.current_gaze_start is None:
                self.current_gaze_start = dt
            self.missed_so_far = 0
        else:
            # No gaze
            self.update_without_face_detection(dt)

    def update_without_face_detection(self, dt: datetime):
        # We only care if we have started gazing
        if self.current_gaze_start is not None:
            if self.missed_so_far >= self.lost_buffer:
                # Should check if its the longest gaze
                gaze_time = dt - self.current_gaze_start
                if self.longest_gaze_so_far is not None:
                    self.longest_gaze_so_far = max(
                        self.longest_gaze_so_far, gaze_time.total_seconds()
                    )
                else:
                    self.longest_gaze_so_far = gaze_time.total_seconds()
                self.current_gaze_start = None
                self.missed_so_far = 0
            self.missed_so_far += 1

    def get(self, dt: t.Optional[datetime]) -> t.Optional[float]:
        """
        dt: is optional as it is used to calculate the final gaze time
            if dt is not given, it will return the longest gaze time so far
        """
        if dt is not None and self.current_gaze_start is not None:
            gaze_time = dt - self.current_gaze_start
            if self.longest_gaze_so_far is not None:
                self.longest_gaze_so_far = max(
                    self.longest_gaze_so_far, gaze_time.total_seconds()
                )
            else:
                self.longest_gaze_so_far = gaze_time.total_seconds()

        return self.longest_gaze_so_far


class FaceImgMetadataProcessor(FaceDetectionMetadataProcessor):
    def __init__(self, config):
        super().__init__(config)
        self.best_face_img: t.Optional[np.ndarray] = None
        self.metadata_metric: t.Optional[MetadataMetric] = None

        self.accurate_face_threshold_config = (
            self.config.gaze.tracker.accurate_face_threshold
        )
        self.face_expand_ratio = self.config.gaze.tracker.face.face_expand_ratio

    def update(self, face: FaceDetection, img):
        """
        img: cv2 RGB
        """
        # Take the first image
        if self.best_face_img is None:
            self.set_face_img(face, img)
            return

        # If we cross the threshold, we care about better angles
        if (
            self.metadata_metric.score > self.accurate_face_threshold_config.score
            and face.score > self.accurate_face_threshold_config.score
        ):
            if abs(face.yaw) + abs(face.pitch) < abs(self.metadata_metric.yaw) + abs(
                self.metadata_metric.pitch
            ):
                self.set_face_img(face, img)
                return

        if face.score > self.metadata_metric.score:
            self.set_face_img(face, img)

    def set_face_img(self, face: FaceDetection, img):
        """
        helper function extract the face_img
        """
        self.best_face_img = get_face_img(face.bbox, img, self.face_expand_ratio)
        if self.best_face_img is not None:
            self.metadata_metric = MetadataMetric.from_face_detection(face)
        return

    def get(self) -> t.Optional[np.ndarray]:
        return self.best_face_img


class FaceReIDMetadataProcessor(FaceDetectionMetadataProcessor):
    def __init__(
        self,
        config,
        face_embedding_model: ArcFace,
        face_landmark_model: MobileNetV2FaceLandmark,
        face_reid_database: FaceReIDBaseDatabase,
    ):
        super().__init__(config)
        self.reid: t.Optional[int] = None

        self.face_embedding_model = face_embedding_model
        self.face_landmark_model = face_landmark_model
        self.face_reid_database = face_reid_database

        self.accurate_face_threshold_config = (
            self.config.gaze.tracker.accurate_face_threshold
        )
        self.face_expand_ratio = self.config.gaze.tracker.face_reid.face_expand_ratio

    def update(self, face: FaceDetection, img):
        """
        img: cv2 RGB
        """
        # Skip if not enabled, or already got a reid
        if not self.config.gaze.tracker.face_reid.enable:
            return
        if self.reid is not None:
            return

        if face.score > self.accurate_face_threshold_config.score:
            # NOTE: ~50-60ms
            face_img = get_face_img(face.bbox, img, self.face_expand_ratio)
            if face_img is None:
                return

            landmarks = self.face_landmark_model.forward(face_img, face.bbox)
            adjusted_face_img = norm_crop_image(face_img, np.array(landmarks))
            embedding = self.face_embedding_model.raw_forward(
                adjusted_face_img
            )  # (1, 512)
            reid = self.face_reid_database.face_reid_entry(embedding, adjusted_face_img)
            if reid is None:
                # reid failed, just ignore then
                return

            self.reid = reid[0]

    def get(self) -> t.Optional[int]:
        return self.reid
