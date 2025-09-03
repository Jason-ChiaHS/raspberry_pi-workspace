from math import cos, sin

import cv2
import numpy as np


def calculate_iou(bbox1, bbox2):
    """

    Compute IoU for two boxes

    """

    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    area1 = max(0.0, (bbox1[2] - bbox1[0])) * max(0.0, (bbox1[3] - bbox1[1]))
    area2 = max(0.0, (bbox2[2] - bbox2[0])) * max(0.0, (bbox2[3] - bbox2[1]))
    union = area1 + area2 - inter_area

    return inter_area / union if union > 0 else 0.0


def calculate_body_face_association(
    body_bboxes: np.array,
    face_bboxes: np.array,
    iou_weight=0.6,
    align_weight=0.4,
    score_thresh=0.1,
    min_overall_iou=0.05,
    min_area_ratio=0.01,
    max_area_ratio=0.5,
):
    """
    Match each body bbox to the best face bbox with extra geometric constraints:
      - face center must lie inside body bbox
      - overall IoU(body,face) >= min_overall_iou
      - face_area / body_area in [min_area_ratio, max_area_ratio]
    Returns list of (body_idx, face_idx).
    """

    associations = []
    assigned_faces = set()

    for body_idx, body in enumerate(body_bboxes):
        x0, y0, x1, y1 = body["bbox"]
        body_cx = (x0 + x1) / 2
        body_cy = (y0 + y1) / 2
        top_half = [x0, y0, x1, (y0 + y1) / 2]
        body_area = max(0.0, (x1 - x0)) * max(0.0, (y1 - y0))
        best_face_idx, best_score = -1, score_thresh

        for face_idx, face in enumerate(face_bboxes):
            if face_idx in assigned_faces:
                continue

            fx0, fy0, fx1, fy1 = face["bbox"]
            fcx = (fx0 + fx1) / 2
            fcy = (fy0 + fy1) / 2

            # 1. face center must be inside body bbox
            if not (x0 <= fcx <= x1 and y0 <= fcy <= y1):
                continue

            # 2. overall IoU constraint
            overall_iou = calculate_iou([x0, y0, x1, y1], [fx0, fy0, fx1, fy1])

            if overall_iou < min_overall_iou:
                continue

            # 3. area ratio constraint
            face_area = max(0.0, (fx1 - fx0)) * max(0.0, (fy1 - fy0))
            ratio = face_area / (body_area + 1e-6)
            if not (min_area_ratio <= ratio <= max_area_ratio):
                continue

            # 4. face must be in upper half (fcy <= body_cy)
            if fcy > body_cy:
                continue

            # IoU on top-half region
            spatial_score = calculate_iou(top_half, [fx0, fy0, fx1, fy1])

            # horizontal alignment score [0,1]
            align = 1 - min(1, abs(fcx - body_cx) / (x1 - x0 + 1e-6))

            # fused score
            score = iou_weight * spatial_score + align_weight * align
            if score > best_score:
                best_score, best_face_idx = score, face_idx

        if best_face_idx >= 0:
            associations.append((body_idx, best_face_idx))
            assigned_faces.add(best_face_idx)

    return associations


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

    if tdx is not None and tdy is not None:
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
        # base_len = math.sqrt((face_x - x3) ** 2 + (face_y - y3) ** 2)
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
