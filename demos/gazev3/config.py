from dataclasses import dataclass, field
from enum import Enum


class DB(Enum):
    MEMORY = "memory"
    SQLITE = "sqlite"
    HTTP_CENTRAL_SERVER = "http_central_server"


@dataclass
class HTTPCentralClient:
    base_url: str = "http://localhost:5500"


@dataclass
class HTTPCentralServer:
    host_port: int = 5500


@dataclass
class Pitch:
    min: int = -35
    max: int = 35


@dataclass
class Yaw:
    min: int = -35
    max: int = 35


@dataclass
class AccurateFaceThreshold:
    pitch: Pitch = field(default_factory=Pitch)
    yaw: Yaw = field(default_factory=Yaw)
    score: float = 0.75


@dataclass
class TrackerGaze:
    # How many frames we can miss the gaze before breaking gaze
    lost_buffer: int = 5  # 0.5s based on 10fps


@dataclass
class AgeEnhancement:
    model_path: str = "./demos/gazev3/weights/mobilenetv4small.onnx"
    face_expand_ratio: float = 0.125
    enable: bool = True


@dataclass
class Age:
    # Min and Max no. of age results to get the average
    min: int = 5
    max: int = 10
    enhancement: AgeEnhancement = field(default_factory=AgeEnhancement)


@dataclass
class Gender:
    # Min and Max no. of gender results to get the average
    min: int = 5
    max: int = 10


@dataclass
class FaceReID:
    enable: bool = True
    face_expand_ratio: float = 0.125
    face_embedding_model_path: str = "./demos/gazev3/weights/w600k_mbf.onnx"
    face_landmark_model_path: str = "./demos/gazev3/weights/landmark_detection_56.onnx"
    similarity_threshold: float = 0.4

    db: str = DB.MEMORY.value


@dataclass
class TrackerFace:
    face_expand_ratio: float = 0.125


@dataclass
class ByteTracker:
    # Should not need to touch
    track_activation_threshold: float = 0.25
    minimum_matching_threshold: float = 0.8
    lost_track_buffer: int = (
        20  # No. of frames a track can be "missing" before considered loast
    )
    frame_rate: int = 10
    # No. of frames a track has to be "seen" before tracking starts
    # 1 is a bit too aggressive, could be higher
    minimum_consecutive_frames: int = 2


@dataclass
class Tracker:
    # config to controls the bytetracker from supervision
    bytetracker: ByteTracker = field(default_factory=ByteTracker)

    # which to decide on tracks metrics/metadta, if a face is accurate
    accurate_face_threshold: AccurateFaceThreshold = field(
        default_factory=AccurateFaceThreshold
    )
    # To calculate gaze time
    gaze: TrackerGaze = field(default_factory=TrackerGaze)
    age: Age = field(default_factory=Age)
    gender: Gender = field(default_factory=Gender)
    face_reid: FaceReID = field(default_factory=FaceReID)
    # config related to face img saved for the track
    face: TrackerFace = field(default_factory=TrackerFace)
    http_central_client: HTTPCentralClient = field(default_factory=HTTPCentralClient)

    max_historical_buffer: int = 50
    # Will discard all historical tracks with any missing fields
    discard_any_empty_historical_tracks: bool = True

    # Follow DB
    db: str = DB.MEMORY.value
    # Only used when db is set to HTTP_CENTRAL_SERVER
    # Also needs to be a valid path string
    cam_name: str = "default_cam"




@dataclass
class ResizedTrackFace:
    width: int = 100
    height: int = 100


@dataclass
class WebUI:
    resized_track_face: ResizedTrackFace = field(default_factory=ResizedTrackFace)


@dataclass
class Association:
    # Don't really need to update, best params based on testing
    iou_weight: float = 0.6
    align_weight: float = 0.4
    score_thresh: float = 0.05
    min_overall_iou: float = 0.01
    min_area_ratio: float = 0.005
    max_area_ratio: float = 0.7


@dataclass
class GazeConfig:
    score_threshold: float = 0.4
    draw_trace: bool = False  # Misc params

    association: Association = field(default_factory=Association)
    tracker: Tracker = field(default_factory=Tracker)
    webui: WebUI = field(default_factory=WebUI)

    http_central_server: HTTPCentralServer = field(default_factory=HTTPCentralServer)
