import base64
from dataclasses import dataclass, field

import cv2
import numpy as np
import supervision as sv
from supervision.annotators.utils import resolve_color
from supervision.detection.core import Detections
from supervision.draw.color import ColorPalette

from sdk.helpers.config import Config
from sdk.helpers.logger import logger

from . import BasePipeline
from .metadata_processor import GazePeopleProcessor, HistoricalDetectionsProcessor
from .models import MobileNet
from .custom_models.mobilenet_face_landmark import MobileNetV2FaceLandmark
from .reid_models.arcface import ArcFace
from .reid_models.scrfd import SCRFD
from .track_utils import Sort
from .utils import (
    calculate_body_face_association,
    plot_3axis_Zaxis,
    update_person_age_gender,
)


@dataclass
class GazeSortConfig:
    max_age: int = 30
    min_hits: int = 2
    iou_threshold: float = 0.3
    max_distance: int = 1014  # 0.5 * max(width, height)
    body_bbox_min_size: int = 30


@dataclass
class GazeFaceConfig:
    min_size: int = 70
    out_margin: int = 50
    pitch_threshold: int = 90
    yaw_threshold: int = 45


@dataclass
class GazeAssociationConfig:
    iou_weight: float = 0.6
    align_weight: float = 0.4
    score_thresh: float = 0.1
    min_overall_iou: float = 0.05
    min_area_ratio: float = 0.01
    max_area_ratio: float = 0.5


@dataclass
class GazeHistoricalDetectionsConfig:
    db_path: str = "./gazev2_historical_detections.db"
    face_imgs_folder_path: str = "./gazev2_face_imgs"


@dataclass
class GazeConfig:
    score_threshold: float = 0.6
    fixed_track_face_preview: bool = False

    cv2_track_face_width: int = 100
    cv2_track_face_height: int = 100

    sort: GazeSortConfig = field(default_factory=GazeSortConfig)
    face: GazeFaceConfig = field(default_factory=GazeFaceConfig)
    association: GazeAssociationConfig = field(default_factory=GazeAssociationConfig)
    historical_detections: GazeHistoricalDetectionsConfig = field(
        default_factory=GazeHistoricalDetectionsConfig
    )

    age_enhancement_model_path: str = "./demos/gaze-v2/models/mobilenetv4small.onnx"
    skip_age_enhancement: bool = False

    face_expand_ratio: float = 0.125

    # TEMP
    use_sv: bool = True


# Implementing the AdditionalDemoConfig is not strictly needed
# Its purpose is to set default values and help with typing of any added vars
@dataclass
class AdditionalDemoConfig(Config):
    gaze: GazeConfig = field(default_factory=GazeConfig)


# Override the methods below, referring to their documentation
# Refer to sdk/base_pipeline.py BasePipeline for how they are used in the respective scripts
class DemoPipeline(BasePipeline):
    def __init__(self, config, cam):
        # Just extend __init__
        super().__init__(config, cam)
        self.config: AdditionalDemoConfig = self.merge_demo_pipeline_config(
            AdditionalDemoConfig()
        )
        self.score_threshold = self.config.gaze.score_threshold

        self.gaze_people_processor = GazePeopleProcessor()
        self.historical_detection_processor = HistoricalDetectionsProcessor(self.config)

        sort_config = self.config.gaze.sort
        self.sort = Sort(
            sort_config.max_age,
            sort_config.min_hits,
            sort_config.iou_threshold,
            sort_config.max_distance,
        )

        self.age_predictions = {}  # { tid: [ages] }
        self.gender_predictions = {}  # { tid: [genders] }
        self.final_age_gender = {}  # { tid: (final_age, final_gender) }
        self.face_img_buffers = {}  # { tid: {face_img: base64_str, count} }

        self.mobilenet = MobileNet(self.config.gaze.age_enhancement_model_path)

        # Supervision
        self.sv_tracker = sv.ByteTrack(
            frame_rate=10, minimum_consecutive_frames=5, lost_track_buffer=20
        )
        self.sv_box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
        self.sv_label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK)
        self.sv_trace_annotator = sv.TraceAnnotator(color_lookup=sv.ColorLookup.TRACK)

        self.sv_frame_counter = 0

        self.scrfd_model = SCRFD(
            "./demos/gaze-v2/models/det_500m.onnx", input_size=(640, 640)
        )
        self.arcface_model = ArcFace("./demos/gaze-v2/models/w600k_mbf.onnx")
        self.face_landmark_model = MobileNetV2FaceLandmark(
            "./demos/gaze-v2/models/landmark_detection_56.onnx"
        )

    def post_processing(self, inference_frame_time, np_outputs, img, cam_metadata):
        face_detections, body_detections = self.process_gaze_v2_model_outputs(
            np_outputs, cam_metadata, img
        )
        for face_detection in face_detections:
            bbox = face_detection["bbox"]
            (x1, y1, x2, y2) = bbox
            face_img = img[y1:y2, x1:x2]
            landmarks = self.face_landmark_model.forward(face_img, bbox)

        if not self.config.gaze.use_sv:
            results = self.tracking_post_processing(face_detections, body_detections)
            self.gaze_people_processor.update_gaze_people_datas(
                results["results"], inference_frame_time
            )
            self.historical_detection_processor.update_last_detections(
                self.sort.get_all_tracked_ids(), results["results"]
            )
        else:
            results = self.sv_tracking_post_processing(face_detections, body_detections)
            self.gaze_people_processor.update_gaze_people_datas(
                results["results"], inference_frame_time
            )

            all_track_ids = set()
            for track in self.sv_tracker.tracked_tracks:
                all_track_ids.add(int(track.external_track_id))
            for track in self.sv_tracker.lost_tracks:
                all_track_ids.add(int(track.external_track_id))
            self.historical_detection_processor.update_last_detections(
                list(all_track_ids), results["results"]
            )

        return results

    def pp_reid_results(self, face_detections):
        for face_detection in face_detections:
            face_img = face_detection["face_img"]
            cv2_face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
            _, kpss = self.scrfd_model(cv2_face_img)
            if len(kpss) == 0:
                print("ohno")
            kps = kpss[0]
            pass

    def pp_scrfd(self, np_outputs, img, cam_metadata):
        print()
        cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        bboxes, kpss = self.scrfd_model.detect(cv2_img)

        n_valid = int(np_outputs[4][0])  # Shape: (1), convert to integer
        scores = np_outputs[1]
        boxes = np_outputs[0]
        labels = np_outputs[2]  # Shape: (15)
        model_bboxes = []
        for res_i in range(n_valid):
            # Each index contains 1 detection
            # Either a face or body
            score = scores[res_i]
            bbox = boxes[res_i]
            label = labels[res_i]
            bbox = self.swap_bbox_xy_yx(bbox)  # yx
            bbox = bbox / self.cam.model_height  # Model is 576x576
            bbox = self.cam.imx500.convert_inference_coords(
                bbox, cam_metadata, self.cam.picam2
            )  # [x, y, w, h]
            (
                x1,
                y1,
                w,
                h,
            ) = bbox
            y2 = y1 + h
            x2 = x1 + w
            bbox = [x1, y1, x2, y2]
            if label == 1:
                model_bboxes.append(bbox)

        print(model_bboxes)  # bboxes
        print([bbox for bbox in bboxes])
        # print(kpss.item())

    def process_gaze_v2_model_outputs(self, np_outputs, cam_metadata, img):
        # Extract tensors from model output
        boxes = np_outputs[0]  # Shape: (15, 4)
        scores = np_outputs[1]  # Shape: (15)
        labels = np_outputs[2]  # Shape: (15)
        # Not used
        # indices = np_outputs[3]        # Shape: (15)
        n_valid = int(np_outputs[4][0])  # Shape: (1), convert to integer
        ages = np_outputs[5]  # Shape: (15)
        genders = np_outputs[6]  # Shape: (15)
        headposes = np_outputs[7]  # Shape: (15, 2) - [pitch, yaw]

        body_detections = []
        face_detections = []
        for res_i in range(n_valid):
            # Each index contains 1 detection
            # Either a face or body
            if scores[res_i] >= self.score_threshold:
                score = scores[res_i]
                bbox = boxes[res_i]
                bbox = self.swap_bbox_xy_yx(bbox)  # yx
                bbox = bbox / self.cam.model_height  # Model is 576x576
                bbox = self.cam.imx500.convert_inference_coords(
                    bbox, cam_metadata, self.cam.picam2
                )  # [x, y, w, h]
                (
                    x1,
                    y1,
                    w,
                    h,
                ) = bbox
                y2 = y1 + h
                x2 = x1 + w
                bbox = [x1, y1, x2, y2]
                label = "head" if int(labels[res_i]) == 1 else "body"
                age = ages[res_i][0] * 100
                gender = "male" if genders[res_i][0] < 0.5 else "female"
                headpose = headposes[res_i] * 100

                result = {
                    "score": score.item(),
                    "bbox": bbox,
                    "label": label,
                    "age": int(age.item()),
                    "gender": gender,
                    "yaw": headpose[0].item(),
                    "pitch": headpose[1].item(),
                    "is_frontal": abs(headpose[1].item())
                    < self.config.gaze.face.pitch_threshold
                    and abs(headpose[0].item()) < self.config.gaze.face.yaw_threshold,
                }
                if label == "head":
                    x = x2 - x1
                    y = y2 - y1
                    fy1 = max(int(y1 - self.config.gaze.face_expand_ratio * y), 0)
                    fy2 = min(
                        int(y2 + self.config.gaze.face_expand_ratio * y),
                        len(img[0]) - 1,
                    )
                    fx1 = max(int(x1 - self.config.gaze.face_expand_ratio * x), 0)
                    fx2 = min(
                        int(x2 + self.config.gaze.face_expand_ratio * x),
                        len(img[1] - 1),
                    )
                    face_img = img[
                        fy1:fy2,
                        fx1:fx2,
                    ][:, :, :3]
                    result["face_img"] = face_img.copy()

                    if not self.config.gaze.skip_age_enhancement:
                        # Calculate mobilenet age
                        empty_face = False
                        for dim in face_img.shape:
                            if dim == 0:
                                empty_face = True
                        if empty_face:
                            continue

                        mobilenet_age = self.mobilenet.run_inference(face_img)
                        result["age"] = int(mobilenet_age.item())

                    face_detections.append(result)

                else:
                    body_detections.append(result)

        return face_detections, body_detections

    def sv_tracking_post_processing(self, face_detections, body_detections):
        xyxy = np.array([detection["bbox"] for detection in body_detections])
        confidence = np.array([detection["score"] for detection in body_detections])


        if len(body_detections) == 0:
            detections = Detections.empty()
        else:
            detections = Detections(xyxy, confidence=confidence)
        detections = self.sv_tracker.update_with_detections(detections)
        body_detections = [
            {
                "bbox": detections.xyxy[detection_idx],
                "track_id": detections.tracker_id[detection_idx].item(),
            }
            for detection_idx in range(len(detections))
        ]

        body_face_association = calculate_body_face_association(
            body_detections,
            face_detections,
            iou_weight=self.config.gaze.association.iou_weight,
            align_weight=self.config.gaze.association.align_weight,
            score_thresh=self.config.gaze.association.score_thresh,
            min_overall_iou=self.config.gaze.association.min_overall_iou,
            min_area_ratio=self.config.gaze.association.min_area_ratio,
            max_area_ratio=self.config.gaze.association.max_area_ratio,
        )

        # print(f"tracker: {len(detections.xyxy)} match: {len(body_face_association)}")
        body_face_detections = []
        for body_idx, face_idx in body_face_association:
            [x1, y1, x2, y2] = body_detections[body_idx]["bbox"]
            body_face_detection = {
                **face_detections[face_idx],
                "face_bbox": face_detections[face_idx]["bbox"],
                "body_bbox": [x1.item(), y1.item(), x2.item(), y2.item()],
                "track_id": body_detections[body_idx]["track_id"],
                "final_age": face_detections[face_idx]["age"],
                "final_gender": face_detections[face_idx]["gender"],
            }
            del body_face_detection["bbox"]
            del body_face_detection["label"]
            body_face_detections.append(body_face_detection)

        untracked_tracks = [
            body_detections[body_idx]
            for body_idx in set(range(len(body_detections))).difference(
                set([body_idx for body_idx, _ in body_face_association])
            )
        ]
        return {"results": body_face_detections, "untracked_tracks": untracked_tracks}

    def tracking_post_processing(self, face_detections, body_detections):
        tracks = self.sort.update([result["bbox"] for result in body_detections])

        # Creating the track_body_bboxes
        track_body_bboxes = []
        for track in tracks:
            x1, y1, x2, y2, tid = track.astype(np.int32)
            tid = int(tid)
            track_body_bboxes.append(
                {"bbox": [x1.item(), y1.item(), x2.item(), y2.item()], "track_id": tid}
            )

        track_face_associations = calculate_body_face_association(
            track_body_bboxes,
            face_detections,
            iou_weight=self.config.gaze.association.iou_weight,
            align_weight=self.config.gaze.association.align_weight,
            score_thresh=self.config.gaze.association.score_thresh,
            min_overall_iou=self.config.gaze.association.min_overall_iou,
            min_area_ratio=self.config.gaze.association.min_area_ratio,
            max_area_ratio=self.config.gaze.association.max_area_ratio,
        )
        body_face_association = calculate_body_face_association(
            body_detections,
            face_detections,
            iou_weight=self.config.gaze.association.iou_weight,
            align_weight=self.config.gaze.association.align_weight,
            score_thresh=self.config.gaze.association.score_thresh,
            min_overall_iou=self.config.gaze.association.min_overall_iou,
            min_area_ratio=self.config.gaze.association.min_area_ratio,
            max_area_ratio=self.config.gaze.association.max_area_ratio,
        )

        body_face_detections = []
        for body_idx, face_idx in body_face_association:
            body_face_detection = {
                **face_detections[face_idx],
                "face_bbox": face_detections[face_idx]["bbox"],
                "body_bbox": body_detections[body_idx]["bbox"],
            }
            del body_face_detection["bbox"]
            del body_face_detection["label"]
            body_face_detections.append(body_face_detection)

        # Find all the tracks with the corrosponding face and body
        face_body_track_mapping = {}
        for track_idx, face_idx in track_face_associations:
            face_body_track_mapping[face_idx] = {
                "track_id": track_body_bboxes[track_idx]["track_id"],
                "track_idx": track_idx,
                **face_detections[face_idx],
                "face_bbox": face_detections[face_idx]["bbox"],
            }
            del face_body_track_mapping[face_idx]["label"]
            del face_body_track_mapping[face_idx]["bbox"]
        for body_idx, face_idx in body_face_association:
            if face_idx in face_body_track_mapping:
                face_body_track_mapping[face_idx]["body_bbox"] = body_detections[
                    body_idx
                ]["bbox"]

        results = [
            face_body_track
            for face_body_track in face_body_track_mapping.values()
            if "track_id" in face_body_track and "body_bbox" in face_body_track
        ]

        # Find all the tracks without a face/body
        untracked_tracks = [
            track_body_bboxes[track_idx]
            for track_idx in set(range(len(track_body_bboxes))).difference(
                set([face_body_track["track_idx"] for face_body_track in results])
            )
        ]

        # Filter based on configs
        def filter_result(result):
            (x1, y1, x2, y2) = result["body_bbox"]
            body_width = x2 - x1
            body_height = y2 - y1

            (x1, y1, x2, y2) = result["face_bbox"]
            face_width = x2 - x1
            face_height = y2 - y1
            if (
                body_width <= self.config.gaze.sort.body_bbox_min_size
                or body_height <= self.config.gaze.sort.body_bbox_min_size
            ):
                return False

            if (
                face_width <= self.config.gaze.face.min_size
                or face_height <= self.config.gaze.face.min_size
            ):
                return False

            return True

        results = [result for result in results if filter_result(result)]

        for output in results:
            if output["is_frontal"]:
                final_age, final_gender = update_person_age_gender(
                    output["track_id"],
                    output["age"],
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

        return {
            "results": results,
            "untracked_tracks": untracked_tracks,
            "body_face_detections": body_face_detections,
            "body_detections": [
                {**body_detection, "body_bbox": body_detection["bbox"]}
                for body_detection in body_detections
            ],
        }

    def swap_bbox_xy_yx(self, bbox):
        return bbox[[1, 0, 3, 2]]

    def draw_metadata(self, inference_frame_time, metadata, img):
        return self.custom_cv2_draw_metadata(metadata, img)
        # return self.sv_draw_metadata(metadata, img)

    def sv_draw_metadata(self, metadata, img):
        results = metadata["results"]
        # Skip if no detections
        if len(results) == 0:
            return img

        xyxy = np.array([result["body_bbox"] for result in results])
        confidence = np.array([result["score"] for result in results])
        tracker_id = np.array([result["track_id"] for result in results])

        detections = Detections(xyxy, confidence=confidence, tracker_id=tracker_id)

        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        annotated_frame = self.sv_box_annotator.annotate(img, detections=detections)
        annotated_frame = self.sv_label_annotator.annotate(
            annotated_frame, detections=detections, labels=labels
        )
        annotated_frame = self.sv_trace_annotator.annotate(
            annotated_frame, detections=detections
        )
        img = annotated_frame

        # Draw face
        for i, result in enumerate(results):
            pitch = result["pitch"]
            yaw = result["yaw"]
            # Based on source code from https://github.com/roboflow/supervision/blob/develop/supervision/annotators/utils.py
            # and https://github.com/roboflow/supervision/blob/develop/supervision/annotators/core.py
            # Which they used to determine the color for the track
            color = resolve_color(
                ColorPalette.DEFAULT, detections, i, sv.ColorLookup.TRACK
            ).as_bgr()
            (
                x1,
                y1,
                x2,
                y2,
            ) = result["face_bbox"]
            # Draw headpose
            img = plot_3axis_Zaxis(
                img,
                yaw,
                pitch,
                0,
                x1 + (x2 - x1) / 2,
                y1 + (y2 - y1) / 2,
                size=50.0,
                limited=True,
                thickness=2,
                extending=False,
            )
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        return img

    def custom_cv2_draw_metadata(self, metadata, img):
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

        results = metadata["results"]
        for result in results:
            age = result["age"]
            gender = result["gender"]
            pitch = result["pitch"]
            yaw = result["yaw"]
            track_id = result["track_id"]

            (
                x1,
                y1,
                x2,
                y2,
            ) = result["body_bbox"]
            color = colors[track_id % len(colors)]
            text = f"TrackID: {track_id}, Age: {age}, Gender: {gender}, Pitch: {pitch}, Yaw: {yaw}"
            # text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            # img = cv2.rectangle(
            #     img,
            #     (x1, y1 - text_size[1] - 10),
            #     (x1 + text_size[0], y1),
            #     color,
            #     -1,
            # )
            # img = cv2.putText(
            #     img,
            #     text,
            #     (x1, y1 - 5),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.7,
            #     (255, 255, 255),
            #     2,
            # )

            # Drawing Face
            (
                x1,
                y1,
                x2,
                y2,
            ) = result["face_bbox"]
            # Draw headpose
            img = plot_3axis_Zaxis(
                img,
                yaw,
                pitch,
                0,
                x1 + (x2 - x1) / 2,
                y1 + (y2 - y1) / 2,
                size=50.0,
                limited=True,
                thickness=2,
                extending=False,
            )
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        for body in metadata["untracked_tracks"]:
            (
                x1,
                y1,
                x2,
                y2,
            ) = body["bbox"]
            track_id = body["track_id"]
            color = colors[track_id % len(colors)]

            # text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            # img = cv2.rectangle(
            #     img,
            #     (x1, y1 - text_size[1] - 10),
            #     (x1 + text_size[0], y1),
            #     color,
            #     -1,
            # )
            # img = cv2.putText(
            #     img,
            #     text,
            #     (x1, y1 - 5),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.7,
            #     (255, 255, 255),
            #     2,
            # )

        return img

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

    def serialize_metadata(self, inference_frame_time, metadata):
        return {
            "current_frame_data": {
                "people_list": [
                    {**result, "face_img": self.b64_encode_face(result["face_img"])}
                    for result in metadata["results"]
                ],
                "gaze": self.gaze_people_processor.latest_gaze_people_data.to_dict(),
                "tracks": [
                    {
                        **self.historical_detection_processor.seen_detections[track_id][
                            "metadata"
                        ],
                        "face_img": self.b64_encode_face(
                            self.historical_detection_processor.seen_detections[
                                track_id
                            ]["metadata"]["face_img"]
                        ),
                        "track_id": track_id,
                        "reid": self.historical_detection_processor.seen_detections[
                            track_id
                        ]["reid"]
                        if "reid"
                        in self.historical_detection_processor.seen_detections[track_id]
                        else -1,
                    }
                    for track_id in self.historical_detection_processor.seen_detections
                ],
            },
            "historical_data": {
                "gaze": self.gaze_people_processor.gaze_people_datas_to_dict(),
                "tracks": [
                    {
                        **last_detection.to_dict(),
                        "face_img": self.b64_encode_face(last_detection.face_img),
                    }
                    for last_detection in self.historical_detection_processor.last_detections
                ],
            },
        }

    def setup_trigger_condition(self):
        super().setup_trigger_condition()
        self.trigger_frame_count = 0
        self.trigger_count = 0

    def update_trigger_condition(self, inference_frame_time, metadata, img):
        self.trigger_frame_count += 1
        if self.trigger_frame_count > 10:
            self.trigger_condition = True

    def reset_trigger_condition(self):
        super().reset_trigger_condition()
        self.trigger_frame_count = 0
        self.trigger_count += 1

    def serialize_trigger_metadata(self, inference_frame_time, metadata):
        return {"trigger_count": self.trigger_count}
