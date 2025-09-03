import typing as t
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel

from sdk.helpers.config import Config

from .config import GazeConfig


# Implementing the AdditionalDemoConfig is not strictly needed
# Its purpose is to set default values and help with typing of any added vars
@dataclass
class AdditionalDemoConfig(Config):
    gaze: GazeConfig = field(default=GazeConfig)


@dataclass
class FaceDetection:
    score: float
    bbox: np.ndarray  # [x1, y1, x2, y2]
    age: int
    gender: t.Literal["Male", "Female"]
    yaw: float
    pitch: float


@dataclass
class BodyDetection:
    score: float
    bbox: np.array  # [x1, y1, x2, y2]


@dataclass
class BodyTrack:
    # Represents a Track with a body detection
    body: BodyDetection
    track_id: int


@dataclass
class FaceBodyTrack(BodyTrack):
    # Represents a Track with body and face detection
    face: FaceDetection


class FaceReIDEntry(BaseModel):
    embedding: list[float]  # [512] because of json encoding
    img: str # base64 encoded

class HistorialDetectionEntry(BaseModel):
    track_id: int
    entry: str # datetime.fromiso
    exit: str # datetime.fromiso
    gaze_time: t.Optional[float]
    gender: t.Optional[str]
    age: t.Optional[float]
    face_reid: t.Optional[int]
    img: t.Optional[str] # base64 encoded
    cam_name: str
