# tracker.py
import math
from math import cos, sin

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def detect_gate_crossing(tracks, gate_line, previous_centroids, direction="vertical"):
    """
    检测人员是否跨越预设的门槛线

    参数：
    - tracks: 当前帧的跟踪结果列表，每个元素为 [x1, y1, x2, y2, track_id]
    - gate_line: 门槛线位置（int）。若为垂直门则为 x 坐标；若为水平门则为 y 坐标
    - previous_centroids: 上一帧中各 track_id 的中心点字典，格式 {track_id: (cx, cy)}
    - direction: 'vertical' 或 'horizontal'，默认为 'vertical'
      - vertical：比较 x 坐标（如从左进入右出）
      - horizontal：比较 y 坐标（如从上进入下出）

    返回：
    - new_centroids: 更新后的中心点字典
    - events: 跨越事件列表，每个元素为 (track_id, event) ，其中 event 为 'in' 或 'out'
    """
    events = []
    new_centroids = {}

    for track in tracks:
        x1, y1, x2, y2, tid = track.astype(np.int32)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        new_centroids[tid] = (cx, cy)

        if tid in previous_centroids:
            prev_cx, prev_cy = previous_centroids[tid]

            if direction == "vertical":
                # 如果之前在门左侧（小于门限）且当前在右侧，则视为进入；反之则为离开
                if prev_cx < gate_line and cx >= gate_line:
                    events.append((tid, "in"))
                elif prev_cx >= gate_line and cx < gate_line:
                    events.append((tid, "out"))
            elif direction == "horizontal":
                # 如果之前在门上方（小于门限）且当前在下方，则视为进入；反之为离开
                if prev_cy < gate_line and cy >= gate_line:
                    events.append((tid, "in"))
                elif prev_cy >= gate_line and cy < gate_line:
                    events.append((tid, "out"))
    return new_centroids, events


def iou(bb_test, bb_gt):
    """
    计算两个边界框的 IoU，边界框格式均为 [x1, y1, x2, y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h

    area1 = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area2 = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])

    if area1 <= 0 or area2 <= 0:
        return 0.0
    return inter / (area1 + area2 - inter + 1e-6)


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox, conf=0.0):
        # 初始化 Kalman 滤波器
        self.kf = cv2.KalmanFilter(7, 4)
        self.kf.transitionMatrix = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],  # cx = cx + vx
                [0, 1, 0, 0, 0, 1, 0],  # cy = cy + vy
                [0, 0, 1, 0, 0, 0, 1],  # s = s + vs
                [0, 0, 0, 1, 0, 0, 0],  # r = r
                [0, 0, 0, 0, 1, 0, 0],  # vx = vx
                [0, 0, 0, 0, 0, 1, 0],  # vy = vy
                [0, 0, 0, 0, 0, 0, 1],  # vs = vs
            ],
            dtype=np.float32,
        )

        self.kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=np.float32,
        )

        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 1e-3
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-2

        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w / 2.0, y1 + h / 2.0
        s, r = w * h, w / float(h) if h > 0 else 1.0

        self.kf.statePre = np.array(
            [[cx], [cy], [s], [r], [0], [0], [0]], dtype=np.float32
        )
        self.kf.statePost = self.kf.statePre.copy()

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits, self.hit_streak, self.age = 0, 0, 0
        self.last_box = bbox
        self.confidence = conf

        # 历史记录，用于平滑预测
        self.history = []
        self.max_history = 5

    def predict(self):
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        cx, cy, s, r = self.kf.statePre[:4, 0]
        s, r = max(s, 1e-6), max(r, 1e-6)
        w, h = np.sqrt(s * r), s / np.sqrt(s * r)
        pred_box = [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]
        pred_box[0] = max(0, pred_box[0])
        pred_box[1] = max(0, pred_box[1])

        self.last_box = pred_box
        return pred_box

    def update(self, bbox, conf=0.0):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.confidence = conf

        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w / 2.0, y1 + h / 2.0
        s, r = w * h, w / float(h) if h > 0 else 1.0

        measurement = np.array([[cx], [cy], [s], [r]], dtype=np.float32)
        self.kf.correct(measurement)

        self.history.append(bbox)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        if len(self.history) >= 3:
            smooth_box = np.mean(self.history[-3:], axis=0)
            self.last_box = smooth_box
        else:
            self.last_box = bbox

    def get_state(self):
        return self.last_box


class Sort:
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3, max_distance=0.7):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_distance = max_distance
        self.trackers = []
        self.frame_count = 0

    def get_all_tracked_ids(self):
        return [track.id for track in self.trackers]

    def update(self, dets, confs=None):
        self.frame_count += 1
        if confs is None:
            confs = np.ones(len(dets))

        trks = (
            np.array([trk.predict() for trk in self.trackers])
            if self.trackers
            else np.empty((0, 4))
        )
        if len(trks) > 0:
            trks = np.maximum(trks, 0)

        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(
            dets, trks
        )

        for m in matched:
            self.trackers[m[1]].update(dets[m[0]], confs[m[0]])

        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[i], confs[i]))

        ret = []
        new_trackers = []
        for i, trk in enumerate(self.trackers):
            if (
                trk.hits >= self.min_hits or self.frame_count <= self.min_hits
            ) and trk.time_since_update <= 1:
                ret.append(
                    np.concatenate((np.array(trk.get_state()), np.array([trk.id])))
                )
            if trk.time_since_update <= self.max_age:
                new_trackers.append(trk)

        self.trackers = new_trackers
        return np.stack(ret) if ret else np.empty((0, 5))

    def associate_detections_to_trackers(self, dets, trks):
        if len(trks) == 0 or len(dets) == 0:
            return (
                np.empty((0, 2), dtype=int),
                np.arange(len(dets)),
                np.arange(len(trks)),
            )

        centers_dets = np.array(
            [[(d[0] + d[2]) / 2.0, (d[1] + d[3]) / 2.0] for d in dets]
        )
        centers_trks = np.array(
            [[(t[0] + t[2]) / 2.0, (t[1] + t[3]) / 2.0] for t in trks]
        )

        sizes_dets = np.array([[(d[2] - d[0]) * (d[3] - d[1])] for d in dets])
        sizes_trks = np.array([[(t[2] - t[0]) * (t[3] - t[1])] for t in trks])

        iou_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
        for d, det in enumerate(dets):
            for t, trk in enumerate(trks):
                iou_matrix[d, t] = iou(det, trk)

        dist_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
        for d, det_center in enumerate(centers_dets):
            for t, trk_center in enumerate(centers_trks):
                dist = np.linalg.norm(det_center - trk_center)
                dist_matrix[d, t] = np.exp(-dist / (self.max_distance * 0.5))

        size_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
        for d in range(len(dets)):
            for t in range(len(trks)):
                ratio = min(sizes_dets[d][0], sizes_trks[t][0]) / max(
                    sizes_dets[d][0], sizes_trks[t][0] + 1e-6
                )
                size_matrix[d, t] = ratio

        combined_matrix = iou_matrix * 0.3 + dist_matrix * 0.6 + size_matrix * 0.1
        combined_matrix = np.nan_to_num(combined_matrix)

        matched_indices = np.array(list(zip(*linear_sum_assignment(-combined_matrix))))

        matches = []
        unmatched_dets = []
        unmatched_trks = []

        for m in matched_indices:
            if combined_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_dets.append(m[0])
                unmatched_trks.append(m[1])
            else:
                matches.append(m)

        for d in range(len(dets)):
            if d not in matched_indices[:, 0]:
                unmatched_dets.append(d)

        for t in range(len(trks)):
            if t not in matched_indices[:, 1]:
                unmatched_trks.append(t)

        return np.array(matches), np.array(unmatched_dets), np.array(unmatched_trks)


def plot_3axis_Zaxis(
    img,
    yaw,
    pitch,
    roll,
    tdx=None,
    tdy=None,
    size=50.0,
    limited=True,
    thickness=2,
    extending=False,
):
    # Input is a cv2 image
    # pose_params: (pitch, yaw, roll, tdx, tdy)
    # Where (tdx, tdy) is the translation of the face.
    # For pose we have [pitch yaw roll tdx tdy tdz scale_factor]

    p = pitch * np.pi / 180
    y = -(yaw * np.pi / 180)
    r = roll * np.pi / 180

    if tdx != None and tdy != None:
        face_x = tdx
        face_y = tdy
    else:
        height, width = img.shape[:2]
        face_x = width / 2
        face_y = height / 2

    # X-Axis (pointing to right) drawn in red
    x1 = size * (cos(y) * cos(r)) + face_x
    y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y

    # Y-Axis (pointing to down) drawn in green
    x2 = size * (-cos(y) * sin(r)) + face_x
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(y)) + face_x
    y3 = size * (-cos(y) * sin(p)) + face_y

    if extending:
        # Plot head oritation extended line in yellow
        # scale_ratio = 5
        scale_ratio = 2
        base_len = math.sqrt((face_x - x3) ** 2 + (face_y - y3) ** 2)
        if face_x == x3:
            endx = tdx
            if face_y < y3:
                if limited:
                    endy = tdy + (y3 - face_y) * scale_ratio
                else:
                    endy = img.shape[0]
            else:
                if limited:
                    endy = tdy - (face_y - y3) * scale_ratio
                else:
                    endy = 0
        elif face_x > x3:
            if limited:
                endx = tdx - (face_x - x3) * scale_ratio
                endy = tdy - (face_y - y3) * scale_ratio
            else:
                endx = 0
                endy = tdy - (face_y - y3) / (face_x - x3) * tdx
        else:
            if limited:
                endx = tdx + (x3 - face_x) * scale_ratio
                endy = tdy + (y3 - face_y) * scale_ratio
            else:
                endx = img.shape[1]
                endy = tdy - (face_y - y3) / (face_x - x3) * (tdx - endx)
        # cv2.line(img, (int(tdx), int(tdy)), (int(endx), int(endy)), (0,0,0), 2)
        # cv2.line(img, (int(tdx), int(tdy)), (int(endx), int(endy)), (255,255,0), 2)
        img = cv2.line(
            img, (int(tdx), int(tdy)), (int(endx), int(endy)), (0, 255, 255), thickness
        )

    # X-Axis pointing to right. drawn in red
    img = cv2.line(
        img, (int(face_x), int(face_y)), (int(x1), int(y1)), (0, 0, 255), thickness
    )
    # Y-Axis pointing to down. drawn in green
    img = cv2.line(
        img, (int(face_x), int(face_y)), (int(x2), int(y2)), (0, 255, 0), thickness
    )
    # Z-Axis (out of the screen) drawn in blue
    img = cv2.line(
        img, (int(face_x), int(face_y)), (int(x3), int(y3)), (255, 0, 0), thickness
    )
    return img
