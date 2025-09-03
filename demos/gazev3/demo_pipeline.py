import typing as t
import cv2
import base64
import numpy as np
import supervision as sv
from supervision.detection.core import Detections

from sdk.base_pipeline import BasePipeline, BaseScript
from .utils import calculate_body_face_association, plot_3axis_Zaxis
from .common import (
    BodyDetection,
    BodyTrack,
    FaceBodyTrack,
    FaceDetection,
    AdditionalDemoConfig,
)
from .processors.tracks import TracksProcessor, HistoricalProcessedTrack, ProcessedTrack
from .http_central_server import HTTPCentralServer


class DemoScript(BaseScript):
    def __init__(self, config):
        super().__init__(config)
        self.config: AdditionalDemoConfig = self.merge_demo_pipeline_config(
            AdditionalDemoConfig()
        )

    def start(self):
        if self.config.custom_script is None:
            return
        if self.config.custom_script == "http_central_server":
            HTTPCentralServer(self.config)
        return super().start()


# Override the methods below, referring to their documentation
# Refer to sdk/base_pipeline.py BasePipeline for how they are used in the respective scripts
class DemoPipeline(BasePipeline):
    def __init__(self, config, cam):
        # Just extend __init__
        super().__init__(config, cam)
        self.config: AdditionalDemoConfig = self.merge_demo_pipeline_config(
            AdditionalDemoConfig()
        )

        byte_tracker_config = self.config.gaze.tracker.bytetracker
        self.sv_tracker = sv.ByteTrack(
            track_activation_threshold=byte_tracker_config.track_activation_threshold,
            lost_track_buffer=byte_tracker_config.lost_track_buffer,
            minimum_matching_threshold=byte_tracker_config.minimum_matching_threshold,
            frame_rate=byte_tracker_config.frame_rate,
            minimum_consecutive_frames=byte_tracker_config.minimum_consecutive_frames,
        )

        self.track_colors = [
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

        if self.config.gaze.draw_trace:
            sv_color_palette = sv.ColorPalette(
                [
                    sv.Color(b=track_color[0], g=track_color[1], r=track_color[2])
                    for track_color in self.track_colors
                ]
            )
            self.sv_trace_annotator = sv.TraceAnnotator(
                color_lookup=sv.ColorLookup.TRACK, color=sv_color_palette
            )

        self.tracks_processor = TracksProcessor(self.config)

    def post_processing(self, inference_frame_time, np_outputs, img, cam_metadata):
        face_detections, body_detections = self.process_gaze_v2_model_outputs(
            np_outputs, cam_metadata
        )
        body_tracks, face_body_tracks = self.sv_tracking_post_processing(
            face_detections, body_detections
        )
        all_track_ids = set(
            map(
                lambda track: int(track.external_track_id),
                self.sv_tracker.tracked_tracks,
            )
        ).union(
            set(
                map(
                    lambda track: int(track.external_track_id),
                    self.sv_tracker.lost_tracks,
                )
            )
        )
        if -1 in all_track_ids:
            # -1 id means it has not started tracking yet
            all_track_ids.remove(-1)
        self.tracks_processor.per_frame_update(all_track_ids, face_body_tracks, img)

        return (body_tracks, face_body_tracks)

    def process_gaze_v2_model_outputs(
        self, np_outputs, cam_metadata
    ) -> tuple[list[FaceDetection], list[BodyDetection]]:
        """
        Returns face_detections, body_detections
        """
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

        body_detections: list[BodyDetection] = []
        face_detections: list[FaceDetection] = []
        for res_i in range(n_valid):
            # Each index contains 1 detection
            # Either a face or body
            if scores[res_i] >= self.config.gaze.score_threshold:
                score = scores[res_i]
                bbox = boxes[res_i]
                bbox = bbox[[1, 0, 3, 2]]  # swap xyxy to yxyx
                bbox = np.array(
                    [
                        bbox[0] / self.cam.model_height,
                        bbox[1] / self.cam.model_width,
                        bbox[2] / self.cam.model_height,
                        bbox[3] / self.cam.model_width,
                    ]
                )
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
                gender = "male" if genders[res_i][0] < 0.35 else "female"
                headpose = headposes[res_i] * 100

                if label == "head":
                    result = FaceDetection(
                        score.item(),
                        bbox,
                        int(age.item()),
                        gender,
                        headpose[0].item(),
                        headpose[1].item(),
                    )
                    face_detections.append(result)
                else:
                    result = BodyDetection(
                        score.item(),
                        bbox,
                    )
                    body_detections.append(result)

        return face_detections, body_detections

    def sv_tracking_post_processing(
        self, face_detections: list[FaceDetection], body_detections: list[BodyDetection]
    ) -> tuple[list[BodyTrack], list[FaceBodyTrack]]:
        """
        Returns a tuple of (body_tracks, face_body_tracks)
        The body_tracks are body that have track_ids, but no associated face on this current frame
        """

        xyxy = np.array([detection.bbox for detection in body_detections])
        confidence = np.array([detection.score for detection in body_detections])

        if len(body_detections) == 0:
            detections = Detections.empty()
        else:
            detections = Detections(xyxy, confidence=confidence)
        detections = self.sv_tracker.update_with_detections(detections)
        # Format for util func
        body_bbox_with_track_ids = [
            {
                "bbox": detections.xyxy[detection_idx],
                "score": detections.confidence[detection_idx],
                "track_id": detections.tracker_id[detection_idx].item(),
            }
            for detection_idx in range(len(detections))
        ]
        face_bboxes = [{"bbox": detection.bbox} for detection in face_detections]

        association_config = self.config.gaze.association
        body_face_association = calculate_body_face_association(
            body_bbox_with_track_ids,
            face_bboxes,
            iou_weight=association_config.iou_weight,
            align_weight=association_config.align_weight,
            score_thresh=association_config.score_thresh,
            min_overall_iou=association_config.min_overall_iou,
            min_area_ratio=association_config.min_area_ratio,
            max_area_ratio=association_config.max_area_ratio,
        )

        body_face_tracks = []
        for body_idx, face_idx in body_face_association:
            body_bbox_with_track_id = body_bbox_with_track_ids[body_idx]
            body_face_tracks.append(
                FaceBodyTrack(
                    BodyDetection(
                        body_bbox_with_track_id["score"],
                        body_bbox_with_track_id["bbox"],
                    ),
                    body_bbox_with_track_id["track_id"],
                    face_detections[face_idx],
                )
            )

        body_tracks = [
            BodyTrack(
                BodyDetection(
                    body_bbox_with_track_ids[body_idx]["score"],
                    body_bbox_with_track_ids[body_idx]["bbox"],
                ),
                body_bbox_with_track_ids[body_idx]["track_id"],
            )
            for body_idx in set(range(len(body_bbox_with_track_ids))).difference(
                set([body_idx for body_idx, _ in body_face_association])
            )
        ]
        return (body_tracks, body_face_tracks)

    def draw_metadata(
        self,
        inference_frame_time,
        metadata: tuple[list[BodyTrack], list[FaceBodyTrack]],
        img,
    ):
        body_tracks, face_body_tracks = metadata
        track_colors = [
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
        for face_body_track in face_body_tracks:
            # Draw body
            bbox = face_body_track.body.bbox
            (x1, y1, x2, y2) = [int(cord) for cord in bbox]
            color = track_colors[face_body_track.track_id % len(track_colors)]
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Draw Face
            bbox = face_body_track.face.bbox
            (x1, y1, x2, y2) = [int(cord) for cord in bbox]
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            img = plot_3axis_Zaxis(
                img,
                face_body_track.face.yaw,
                face_body_track.face.pitch,
                0,
                x1 + (x2 - x1) / 2,
                y1 + (y2 - y1) / 2,
                size=50.0,
                limited=True,
                thickness=2,
                extending=False,
            )

        for body_track in body_tracks:
            bbox = body_track.body.bbox
            (x1, y1, x2, y2) = [int(cord) for cord in bbox]
            color = track_colors[body_track.track_id % len(track_colors)]
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw trace for all
        if self.config.gaze.draw_trace:
            xyxy = []
            tracker_id = []
            for track in [*body_tracks, *face_body_tracks]:
                track = t.cast(BodyTrack, track)  # Base class
                bbox = track.body.bbox
                track_id = track.track_id
                xyxy.append(bbox)
                tracker_id.append(track_id)
            if len(xyxy) > 0:
                detections = Detections(np.array(xyxy), tracker_id=np.array(tracker_id))
                img = self.sv_trace_annotator.annotate(img, detections)

        return img

    def b64_encode_face(self, face: np.ndarray):
        """
        Takes in face as RGB since the original frame is in RGB

        Also returns the encoded_image as a jpg string
        """
        if not self.config.upload_metadata.send_frame:
            return ""
        img = face
        b64_resize_width = self.config.gaze.webui.resized_track_face.width
        b64_resize_height = self.config.gaze.webui.resized_track_face.height
        img = cv2.resize(img, (b64_resize_width, b64_resize_height))  # Resize image
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _, encoded_image = cv2.imencode(".jpg", img)
        frame = base64.b64encode(encoded_image.tobytes()).decode("utf-8")
        return frame

    def serialize_processed_track(self, track: ProcessedTrack):
        age = track.age.get()
        gender = track.gender.get()
        face_img = track.face_img.get()
        face_reid = track.face_reid.get()
        return {
            "track_id": track.track_id,
            "entry": track.entry.isoformat(),
            "age": "" if age is None else age,
            "gender": "" if gender is None else gender,
            "face_img": "" if face_img is None else self.b64_encode_face(face_img),
            "pitch": "",
            "yaw": "",
            "score": "",
            "is_frontal": False if track.gaze_time.current_gaze_start is None else True,
            "face_reid": "" if face_reid is None else face_reid,
        }

    def serialize_historical_processed_track(
        self, track: HistoricalProcessedTrack
    ) -> dict:
        """
        For use in the serialize_metadata as the historical_data
        """
        return {
            "track_id": track.track_id,
            "enter_time": track.entry.isoformat(),
            "exit_time": track.exit.isoformat(),
            "longest_gaze": "" if track.gaze_time is None else track.gaze_time,
            "face_img": ""
            if track.face_img is None
            else self.b64_encode_face(track.face_img),
            "gender": "" if track.gender is None else track.gender,
            "age": "" if track.age is None else track.age,
            "gaze_metric": "",
            "score": "",
            "face_reid": "" if track.face_reid is None else track.face_reid,
        }

    def serialize_metadata(self, inference_frame_time, metadata):
        return {
            "current_frame_data": {
                "tracks": [
                    self.serialize_processed_track(track)
                    for track in self.tracks_processor.current_tracks.values()
                ]
            },
            "historical_data": {
                "tracks": [
                    self.serialize_historical_processed_track(past_track)
                    for past_track in self.tracks_processor.past_tracks
                ]
            },
        }
