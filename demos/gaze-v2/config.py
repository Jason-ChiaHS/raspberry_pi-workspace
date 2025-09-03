from dataclasses import dataclass, field

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
    lost_buffer: int = 5 # 0.5s based on 10fps

@dataclass
class AgeEnhancement:
    model_path: str = "./demos/gaze-v2/models/mobilenetv4small.onnx"
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
    model_path: str = "./demos/gaze-v2/models/w600k_mbf.onnx"

@dataclass
class Tracker:
    # Used to increase the size of the face image saved for the track
    face_image_expand_ratio: float = 0.125

    # which to decide on tracks metrics/metadta, if a face is accurate
    accurate_face_threshold: AccurateFaceThreshold = field(default_factory=AccurateFaceThreshold)
    # To calculate gaze time
    gaze: TrackerGaze = field(default_factory=TrackerGaze)
    age: Age = field(default_factory=Age)
    gender: Gender = field(default_factory=Gender)
    face_reid: FaceReID = field(default_factory=FaceReID)

@dataclass
class ResizedTrackFace:
    width: int = 100
    height: int = 100

@dataclass
class WebUI:
    resized_track_face: ResizedTrackFace = field(default_factory=ResizedTrackFace)

@dataclass
class GazeConfig:
    score_threshold: float = 0.4
    trace_preview: bool = False # Misc params

    tracker: Tracker = field(default_factory=Tracker)
    webui: WebUI = field(default_factory=WebUI)
