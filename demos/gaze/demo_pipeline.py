import base64
import json
import typing as t
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from sdk.helpers.config import Config

from . import BasePipeline
from .age_utils import update_person_age_gender
from .gaze_post_processing import GazePostProcessing
from .models import MobileNet
from .track_utils import Sort, detect_gate_crossing, plot_3axis_Zaxis


@dataclass
class GazeSortConfig:
    max_age: int = 30
    min_hits: int = 2
    iou_threshold: float = 0.3
    max_distance: int = 1014  # 0.5 * max(width, height)
    body_bbox_min_size: int = 30


@dataclass
class GazeGateConfig:
    line: int = 520
    direction: str = "horizontal"  # horizontal / vertical


@dataclass
class GazeFaceConfig:
    min_size: int = 70
    out_margin: int = 50


@dataclass
class GazeAgeConfig:
    pitch_threshold: int = 90
    yaw_threshold: int = 45


@dataclass
class GazeConfig:
    mobilenet_model_path: str = "./demos/gaze/models/mobilenetv4small.onnx"
    tflite: bool = True
    face_detection_expand_ratio: float = 0.125
    score_threshold: float = 0.3
    nms_threshold: float = 0.3

    # Fixes the face shown as the preview for the track with the first frame of that given track
    fixed_track_face_preview: bool = True

    cv2_show_body_bbox: bool = True
    cv2_show_face_bbox: bool = True

    cv2_track_face_width: int = 100
    cv2_track_face_height: int = 100

    # Saves everything to allow for offline debugging, will be slow due to waiting on file IO
    # Also will take up a lot of space
    debug_save_overlay_and_metadata: bool = False

    sort: GazeSortConfig = field(default_factory=GazeSortConfig)
    gate: GazeGateConfig = field(default_factory=GazeGateConfig)
    face: GazeFaceConfig = field(default_factory=GazeFaceConfig)
    age: GazeAgeConfig = field(default_factory=GazeAgeConfig)


@dataclass
class AdditionalDemoConfig(Config):
    """
    Extend the config.yaml file in the demo with additional vars
    """

    gaze: GazeConfig = field(default_factory=GazeConfig)


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
    accurate: bool

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
            "accurate": self.accurate,
        }


class MetadataProcessor:
    def __init__(self):
        self.gaze_seconds_interval = 1
        self.gaze_people_datas: list[GazePeopleData] = []
        self.gaze_people_data_ring_size = 20
        self.gaze_people_data_buffer: list[GazePeopleData] = []
        self.latest_gaze_people_data: t.Optional[GazePeopleData] = None

        self.detections_max_size = 500
        self.last_detections: list[LastDetection] = []
        self.seen_detections = {}

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
            if track_id in self.seen_detections:
                if track_id in detections:
                    detection = detections[track_id]
                    self.seen_detections[track_id]["detection_frames"].append(
                        DetectionFrame(dt, detection["is_frontal"])
                    )
                    if not self.seen_detections[track_id]["accurate"]:
                        seen_detection_metadata = self.generate_seen_detection_metadata(
                            detection
                        )
                        if seen_detection_metadata["accurate"]:
                            self.seen_detections[track_id]["metadata"] = (
                                seen_detection_metadata["metadata"]
                            )
                            self.seen_detections[track_id]["accurate"] = (
                                seen_detection_metadata["accurate"]
                            )
                else:
                    self.seen_detections[track_id]["detection_frames"].append(
                        DetectionFrame(dt, False)
                    )

        # Check if we have any seen tracks that are not currently being tracked
        # Remove them and calculate last detection
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
            self.last_detections.append(
                LastDetection(
                    removed_track_id,
                    seen_detection["detection_frames"][0].dt,
                    seen_detection["detection_frames"][-1].dt,
                    longest_gaze_so_far,
                    seen_detection["metadata"]["face_img"],
                    seen_detection["metadata"]["gender"],
                    seen_detection["metadata"]["age"],
                    seen_detection["accurate"],
                )
            )
            del self.seen_detections[removed_track_id]

    def generate_seen_detection_metadata(self, detection):
        # Returns a dict with metadata and accurate
        return {
            "metadata": {
                "age": detection["mobilenet_age"].item()
                if detection["final_age"] is None
                else detection["final_age"],
                "gender": detection["gender"]
                if detection["final_gender"] is None
                else detection["final_gender"],
                "face_img": detection["face_img"],
            },
            "accurate": detection["is_frontal"].item()
            and detection["final_age"] is not None
            and detection["final_gender"] is not None,
        }

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


class DemoPipeline(BasePipeline):
    def __init__(self, config, cam):
        super().__init__(config, cam)
        self.metadata_processor = MetadataProcessor()
        self.config: AdditionalDemoConfig = self.merge_demo_pipeline_config(
            AdditionalDemoConfig()
        )
        self.cv2_single_thread = self.config.single_thread
        try:
            self.cv2_show_body_bbox = self.config.gaze.cv2_show_body_bbox
            self.cv2_show_face_bbox = self.config.gaze.cv2_show_face_bbox
            tflite = self.config.gaze.tflite
            self.gaze_post_processing = GazePostProcessing(
                tflite,
                self.model_width,
                self.model_height,
                self.config.cam.width,
                self.config.cam.height,
                face_detection_expand_ratio=self.config.gaze.face_detection_expand_ratio,
                score_threshold=self.config.gaze.score_threshold,
                nms_threshold=self.config.gaze.nms_threshold,
            )

            mobilenet_model_path = self.config.gaze.mobilenet_model_path
            self.mobilenet = MobileNet(mobilenet_model_path)

            sort_config = self.config.gaze.sort
            self.sort = Sort(
                sort_config.max_age,
                sort_config.min_hits,
                sort_config.iou_threshold,
                sort_config.max_distance,
            )
            self.body_bbox_min_size = sort_config.body_bbox_min_size

            gate_config = self.config.gaze.gate
            self.gate_line = gate_config.line
            self.gate_direction = gate_config.direction

            face_config = self.config.gaze.face
            self.face_min_size = face_config.min_size
            self.face_out_margin = face_config.out_margin

            age_config = self.config.gaze.age
            self.pitch_threshold = age_config.pitch_threshold
            self.yaw_threshold = age_config.yaw_threshold

        except:
            raise Exception("config.toml is misconfigured")

        self.previous_centroids = {}  # 用于存储门口跨越检测所需的中心点

        self.enter_count = 0  # 初始化进入和离开计数器
        self.exit_count = 0

        self.age_predictions = {}  # { tid: [ages] }
        self.gender_predictions = {}  # { tid: [genders] }
        self.final_age_gender = {}  # { tid: (final_age, final_gender) }
        self.face_img_buffers = {}  # { tid: {face_img: base64_str, count} }

    def post_processing(self, inference_frame_time, np_outputs, img, cam_metadata):
        outputs = self.gaze_post_processing.post_processing_tensor(np_outputs)
        outputs = self.gaze_post_processing.scale_results(outputs)
        body_bboxes = []
        new_outputs = []
        for i in range(len(outputs[0])):
            body_bbox = outputs[0][i]
            body_width = body_bbox[2] - body_bbox[0]
            body_height = body_bbox[3] - body_bbox[1]

            face_bbox = outputs[1][i]
            face_width = face_bbox[2] - face_bbox[0]
            face_height = face_bbox[3] - face_bbox[1]
            face_img = img[
                int(face_bbox[1]) : int(face_bbox[3]),
                int(face_bbox[0]) : int(face_bbox[2]),
            ][:, :, :3]
            # We filter based on the minimum body bbox size, for sort but also in general
            if (
                body_width <= self.body_bbox_min_size
                or body_height <= self.body_bbox_min_size
            ):
                continue
            # Filter based on face size
            if face_width <= self.face_min_size or face_height <= self.face_min_size:
                continue

            body_bboxes.append(body_bbox.numpy())

            # Check for an empty face, crashes mobilenet
            empty_face = False
            for dim in face_img.shape:
                if dim == 0:
                    empty_face = True
            if empty_face:
                continue
            mobilenet_age = self.mobilenet.run_inference(face_img)[0]

            metadata = outputs[2][i]
            detection = {
                "body_bbox": body_bbox,
                "face_bbox": face_bbox,
                "metadata": metadata,
                "mobilenet_age": mobilenet_age,
                "gender": "Male" if metadata[0] >= metadata[1] else "Female",
                "is_frontal": abs(metadata[3]) < self.pitch_threshold
                and abs(metadata[4]) < self.yaw_threshold,
                "pitch": metadata[3],
                "yaw": metadata[4],
                "face_img": self.b64_encode_face(face_img),
            }

            new_outputs.append(detection)

        tracks = self.sort.update(body_bboxes)
        # logger.info(f"tracks: {tracks}")

        tids = set()
        for track in tracks:
            x1, y1, x2, y2, tid = track.astype(np.int32)
            tid = int(tid)
            tids.add(tid)
            for output in new_outputs:
                fx1, fy1, fx2, fy2 = output["face_bbox"][:4].tolist()
                if (
                    fx1 >= x1 - self.face_out_margin
                    and fy1 >= y1 - self.face_out_margin
                    and fx2 <= x2 + self.face_out_margin
                    and fy2 <= y2 + self.face_out_margin
                ):
                    output["track_id"] = tid
                    if self.config.gaze.fixed_track_face_preview:
                        if tid in self.face_img_buffers:
                            output["face_img"] = self.face_img_buffers[tid]
                        else:
                            self.face_img_buffers[tid] = output["face_img"]
                    # Ensure no duplicates, taking the first fit to IOU
                    break

        # Memory cleanup for dangling references to old trackids
        for tid in tids.difference(set(self.face_img_buffers.keys())):
            if tid in self.face_img_buffers:
                # Remove the face image from the buffer
                del self.face_img_buffers[tid]

        # Filter out any detections without a trackid
        new_outputs = list(filter(lambda output: "track_id" in output, new_outputs))

        for output in new_outputs:
            if output["is_frontal"]:
                final_age, final_gender = update_person_age_gender(
                    output["track_id"],
                    output["mobilenet_age"],
                    output["gender"],
                    self.age_predictions,
                    self.gender_predictions,
                    self.final_age_gender,
                )
                output["final_age"] = final_age
                output["final_gender"] = final_gender
            else:
                if output["track_id"] in self.final_age_gender:
                    final_age, final_gender = self.final_age_gender[output["track_id"]]
                    output["final_age"] = final_age
                    output["final_gender"] = final_gender
                else:
                    output["final_age"] = None
                    output["final_gender"] = None

        self.metadata_processor.update_gaze_people_datas(
            new_outputs, inference_frame_time
        )
        self.metadata_processor.update_last_detections(
            self.sort.get_all_tracked_ids(), new_outputs
        )
        return (tracks, new_outputs)

    def draw_metadata(self, inference_frame_time, metadata, img):
        (tracks, new_outputs) = metadata
        colors = [
            (0, 255, 0),
            (255, 0, 0),
            (0, 0, 255),
            (255, 255, 0),
            (0, 255, 255),
            (255, 0, 255),
            (128, 0, 0),
            (0, 128, 0),
            (0, 0, 128),
            (128, 128, 0),
        ]

        for output in new_outputs:
            x1, y1, x2, y2 = output["body_bbox"][:4].tolist()
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            tid = output["track_id"]
            color_id = tid % 10
            color = colors[color_id]
            label = f"ID {tid}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            if self.cv2_show_body_bbox:
                img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                img = cv2.rectangle(
                    img,
                    (x1, y1 - text_size[1] - 10),
                    (x1 + text_size[0], y1),
                    color,
                    -1,
                )
                img = cv2.putText(
                    img,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
            fx1, fy1, fx2, fy2 = output["face_bbox"][:4].tolist()
            fx1 = int(fx1)
            fx2 = int(fx2)
            fy1 = int(fy1)
            fy2 = int(fy2)
            if self.cv2_show_face_bbox:
                # Show face regardless of if age is calculated
                fx1, fy1, fx2, fy2 = (
                    int(fx1),
                    int(fy1),
                    int(fx2),
                    int(fy2),
                )
                img = cv2.rectangle(img, (fx1, fy1), (fx2, fy2), color, 2)
                img = cv2.rectangle(
                    img,
                    (fx1, fy1 - text_size[1] - 10),
                    (fx1 + text_size[0], fy1),
                    color,
                    -1,
                )
            if output["is_frontal"]:
                # logger.info(f"age: {self.age_predictions}")
                final_age, final_gender = output["final_age"], output["final_gender"]
                if final_age is not None:
                    pitch = output["pitch"].tolist()
                    yaw = output["yaw"].tolist()
                    info_text = f"Age: {final_age}, Gender: {final_gender}, Pitch: {round(pitch, 2)}, Yaw: {round(yaw, 2)}"
                    # info_text = f"Age: {output['mobilenet_age']}, Gender: {final_gender}, Pitch: {round(pitch, 2)}, Yaw: {round(yaw, 2)}"
                    text_y_pos = y1 - 25 if y1 > 30 else y1 + 20
                    text_size = cv2.getTextSize(
                        info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )[0]
                    # Others
                    img = cv2.putText(
                        img,
                        label,
                        (fx1, fy1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                    img = cv2.rectangle(
                        img,
                        (x1, text_y_pos - text_size[1] - 5),
                        (x1 + text_size[0], text_y_pos + 5),
                        (0, 0, 0),
                        -1,
                    )
                    img = cv2.putText(
                        img,
                        info_text,
                        (x1, text_y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2,
                    )
                    img = plot_3axis_Zaxis(
                        img,
                        yaw,
                        pitch,
                        0,
                        fx1 + (fx2 - fx1) / 2,
                        fy1 + (fy2 - fy1) / 2,
                        size=50.0,
                        limited=True,
                        thickness=2,
                        extending=False,
                    )

        self.previous_centroids, gate_events = detect_gate_crossing(
            tracks, self.gate_line, self.previous_centroids, self.gate_direction
        )
        for tid, event in gate_events:
            if event == "in":
                self.enter_count += 1
            elif event == "out":
                self.exit_count += 1
            # 可选择在目标框附近显示事件信息
            for track in tracks:
                if int(track[4]) == tid:
                    x1, y1, _, _, _ = track.astype(np.int32)
                    img = cv2.putText(
                        img,
                        f"ID {tid} {event}",
                        (x1, y1 - 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

        img = cv2.putText(
            img,
            f"Entered: {self.enter_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        img = cv2.putText(
            img,
            f"Exited: {self.exit_count}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        self.save_overlay_and_metadata(inference_frame_time, metadata, img)
        return img

    def save_overlay_and_metadata(self, inference_frame_time, metadata, img):
        if self.config.gaze.debug_save_overlay_and_metadata:
            artifact_dir = Path(self.config.generate_artifacts.artifact_directory)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            inference_artifact_dir = artifact_dir / str(inference_frame_time)
            inference_artifact_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                (inference_artifact_dir / "drawn.png").as_posix(),
                cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR),
            )
            with open((inference_artifact_dir / "metadata.json"), "w") as f:
                f.write(json.dumps(self.serialize_metadata(metadata)))

    def serialize_metadata(self, inference_frame_time, metadata):
        (tracks, new_outputs) = metadata
        new_tracks = [track.tolist() for track in tracks]
        metadata = {
            "inference_frame_time": inference_frame_time,
            "tracks": new_tracks,
            # "age_predictions": self.age_predictions,
            # "final_age_gender": self.final_age_gender,
            "metadata": [
                {
                    **detection,
                    "body_bbox": detection["body_bbox"].tolist(),
                    "face_bbox": detection["face_bbox"].tolist(),
                    "metadata": detection["metadata"].tolist(),
                    "mobilenet_age": detection["mobilenet_age"].item(),
                    "pitch": detection["pitch"].tolist(),
                    "yaw": detection["yaw"].item(),
                    "is_frontal": detection["is_frontal"].item(),
                }
                for detection in new_outputs
            ],
            "processed_data": {
                "latest_gaze_people_data": self.metadata_processor.latest_gaze_people_data.to_dict(),
                "gaze_people_datas": self.metadata_processor.gaze_people_datas_to_dict(),
                "last_detections": [
                    last_detection.to_dict()
                    for last_detection in self.metadata_processor.last_detections
                ],
            },
        }
        # print(
        #     [
        #         {k: v for (k, v) in detection.items() if k != "face_img"}
        #         for detection in metadata["processed_data"]["last_detections"]
        #     ]
        # )
        return metadata

    def b64_encode_face(self, face):
        """
        For encoding the faces based on trackid
        Takes in face as RGB since the original frame is in RGB

        Also returns the encoded_image as a jpg string
        """
        if not self.config.upload_metadata.send_frame:
            return ""
        img = face
        b64_resize_width = self.config.gaze.cv2_track_face_width
        b64_resize_height = self.config.gaze.cv2_track_face_height
        img = cv2.resize(img, (b64_resize_width, b64_resize_height))  # Resize image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _, encoded_image = cv2.imencode(".jpg", img)
        frame = base64.b64encode(encoded_image.tobytes()).decode("utf-8")
        return frame
