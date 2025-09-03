import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import nms


def squeeze_tensor(tensor: np.ndarray, squeezeN=False):
    """
    Built based on get_valid_tensor_from_dict from model.modules.utils.gt_handler
    """
    if squeezeN:
        while len(tensor.shape) > 3:  # c h w
            tensor = tensor.squeeze(0)
    tensor = torch.from_numpy(tensor)
    return tensor


def get_max_score_coords(center_map, coord):  # 自适应调整H， W
    H, W = center_map.shape[-2], center_map.shape[-1]
    if (
        int(coord[0] + 0.5 - 1) < 0
        or int(coord[0] + 0.5 + 2) > H
        or int(coord[1] + 0.5 - 1) < 0
        or int(coord[1] + 0.5 + 2) > W
    ):
        coord[0] = max(min(int(coord[0] + 0.5), H - 1), 0)
        coord[1] = max(min(int(coord[1] + 0.5), W - 1), 0)
        return coord
    new_map = center_map[
        int(coord[0] + 0.5 - 1) : int(coord[0] + 0.5 + 2),
        int(coord[1] + 0.5 - 1) : int(coord[1] + 0.5 + 2),
    ]

    max_idx = torch.nonzero(new_map == new_map.max())
    coord[0] = max(
        min(int(coord[0] + max_idx[0][0].type_as(coord) + 0.5 - 1.0), H - 1), 0
    )
    coord[1] = max(
        min(int(coord[1] + max_idx[0][1].type_as(coord) + 0.5 - 1.0), W - 1), 0
    )

    return coord


def correct_ind(center_map, coords):
    for coord in coords:
        coord = get_max_score_coords(center_map, coord)
    return coords


class GazePostProcessing:
    def __init__(
        self,
        tflite: bool,
        model_width,
        model_height,
        width,
        height,
        face_detection_expand_ratio: float = 0.125,
        score_threshold=0.3,
        nms_threshold=0.3,
    ):
        self.tflite = tflite
        self.cfg = {
            "kernel_size": 3,
            "score_thr": score_threshold,
            "nms_thr": nms_threshold,
        }
        self.wr = width / model_width
        self.hr = height / model_height
        self.face_detection_expand_ratio = face_detection_expand_ratio

    def draw_bbox(self, image, bbox, color):
        """
        bbox: [x1, y1, x2, y2]
        """
        return cv2.rectangle(
            image,
            (int(bbox[0].item()), int(bbox[1].item())),
            (int(bbox[2].item()), int(bbox[3].item())),
            color,
        )

    def scale_results(self, outputs):
        for i in range(2):
            for ii in range(len(outputs[i])):
                outputs[i][ii][0] = outputs[i][ii][0] * self.wr
                outputs[i][ii][2] = outputs[i][ii][2] * self.wr

                outputs[i][ii][1] = outputs[i][ii][1] * self.hr
                outputs[i][ii][3] = outputs[i][ii][3] * self.hr

        i = 1
        for ii in range(len(outputs[i])):
            x = outputs[i][ii][2] - outputs[i][ii][0]
            outputs[i][ii][0] = outputs[i][ii][0] - x * self.face_detection_expand_ratio
            outputs[i][ii][2] = outputs[i][ii][2] + x * self.face_detection_expand_ratio

            y = outputs[i][ii][3] - outputs[i][ii][1]
            outputs[i][ii][1] = outputs[i][ii][1] - y * self.face_detection_expand_ratio
            outputs[i][ii][3] = outputs[i][ii][3] + y * self.face_detection_expand_ratio
        return outputs

    def post_processing_tensor(
        self,
        np_outputs,
    ):
        cfg = self.cfg

        # Not used, will rescale using another function later
        img_meta = {"scale_factor": 1}
        rescale = False
        feat_stride = 16
        center_name = "main_task_center"

        if not self.tflite:
            # using onnx model
            # C H W, 0 1 2
            center_map = squeeze_tensor(np_outputs[0], squeezeN=True)  # output
            offset_result = squeeze_tensor(np_outputs[1], squeezeN=True)  # 825
            size_result = squeeze_tensor(np_outputs[2], squeezeN=True)  # 853
            subtask_box_center_result = squeeze_tensor(
                np_outputs[6], squeezeN=True
            )  # 890
            subtask_box_offset_result = squeeze_tensor(
                np_outputs[7], squeezeN=True
            )  # 892
            subtask_box_size_result = squeeze_tensor(
                np_outputs[8], squeezeN=True
            )  # 902
            subtask_center_offset_result = squeeze_tensor(
                np_outputs[3], squeezeN=True
            )  # 829
            subtask_face_gender_result = squeeze_tensor(
                np_outputs[5], squeezeN=True
            )  # 833
            # Could be empty??
            subtask_face_age_result = squeeze_tensor(
                np_outputs[9], squeezeN=True
            )  # 904
            subtask_face_headpose_result = squeeze_tensor(
                np_outputs[11], squeezeN=True
            )  # 900
        else:
            # using the tflite model
            # H W C, 1 2 0
            def tt_tensor(t: torch.Tensor):
                t = t.transpose(0, 2)
                t = t.transpose(1, 2)
                return t

            center_map = squeeze_tensor(np_outputs[0], squeezeN=True)  # output
            center_map = tt_tensor(center_map)
            offset_result = squeeze_tensor(np_outputs[1], squeezeN=True)  # 825
            offset_result = tt_tensor(offset_result)
            size_result = squeeze_tensor(np_outputs[2], squeezeN=True)  # 853
            size_result = tt_tensor(size_result)
            subtask_center_offset_result = squeeze_tensor(
                np_outputs[3], squeezeN=True
            )  # 829
            subtask_center_offset_result = tt_tensor(subtask_center_offset_result)
            subtask_face_gender_result = squeeze_tensor(
                np_outputs[4], squeezeN=True
            )  # 833
            subtask_face_gender_result = tt_tensor(subtask_face_gender_result)
            subtask_face_headpose_result = squeeze_tensor(
                np_outputs[10], squeezeN=True
            )  # 900
            subtask_face_headpose_result = tt_tensor(subtask_face_headpose_result)
            subtask_box_center_result = squeeze_tensor(
                np_outputs[6], squeezeN=True
            )  # 890
            subtask_box_center_result = tt_tensor(subtask_box_center_result)
            subtask_box_offset_result = squeeze_tensor(
                np_outputs[7], squeezeN=True
            )  # 892
            subtask_box_offset_result = tt_tensor(subtask_box_offset_result)
            subtask_box_size_result = squeeze_tensor(
                np_outputs[8], squeezeN=True
            )  # 902
            subtask_box_size_result = tt_tensor(subtask_box_size_result)
            # Could be empty??
            subtask_face_age_result = squeeze_tensor(
                np_outputs[9], squeezeN=True
            )  # 904
            subtask_face_age_result = tt_tensor(subtask_face_age_result)

        num_classes = center_map.shape[0]
        det_body_bboxes = torch.tensor([])
        det_face_bboxes = torch.tensor([])
        det_center_offset = torch.tensor([])
        det_face_attributes = torch.tensor([])
        extras_tensors = torch.tensor([])

        for cls_ind in range(num_classes):
            cur_map = center_map[cls_ind, :, :].sigmoid()
            padding = int(cfg["kernel_size"] / 2)
            max_pool = nn.MaxPool2d(cfg["kernel_size"], stride=1, padding=padding)
            max_map = max_pool(cur_map.unsqueeze(0)).squeeze(0)
            is_peak = max_map == cur_map

            subtask_cur_map = subtask_box_center_result[cls_ind, :, :].sigmoid()
            if cfg is not None:
                is_peak *= cur_map >= cfg["score_thr"]

            peak_inds = torch.nonzero(is_peak).type_as(cur_map)
            if peak_inds.numel() == 0:
                continue

            offsets = torch.cat(
                (
                    offset_result[0, :][is_peak].view(-1, 1),
                    offset_result[1, :][is_peak].view(-1, 1),
                ),
                dim=1,
            )
            sizes = torch.cat(
                (
                    size_result[0, :][is_peak].view(-1, 1),
                    size_result[1, :][is_peak].view(-1, 1),
                ),
                dim=1,
            )
            scores = cur_map[is_peak].view(-1, 1)
            sub_task_offsets = torch.cat(
                (
                    subtask_center_offset_result[0, :][is_peak].view(-1, 1),
                    subtask_center_offset_result[1, :][is_peak].view(-1, 1),
                ),
                dim=1,
            )

            sub_peak_inds = peak_inds + sub_task_offsets
            sub_peak_inds = correct_ind(
                subtask_cur_map, sub_peak_inds
            )  # 将直接预测的中心点heatmap 与 bodycenteroffset求出来的中心点 结合在一起得到最终的中心点
            sub_scores = subtask_cur_map[sub_peak_inds.transpose(0, 1).tolist()].view(
                -1, 1
            )
            try:
                sub_task_offsets = torch.cat(
                    (
                        subtask_box_offset_result[0, :][
                            sub_peak_inds.transpose(0, 1).tolist()
                        ].view(-1, 1),
                        subtask_box_offset_result[1, :][
                            sub_peak_inds.transpose(0, 1).tolist()
                        ].view(-1, 1),
                    ),
                    dim=1,
                )
            except:  # noqa: E722
                print("sub_peak_inds ", sub_peak_inds)
                return None
            sub_task_sizes = torch.cat(
                (
                    subtask_box_size_result[0, :][
                        sub_peak_inds.transpose(0, 1).tolist()
                    ].view(-1, 1),
                    subtask_box_size_result[1, :][
                        sub_peak_inds.transpose(0, 1).tolist()
                    ].view(-1, 1),
                ),
                dim=1,
            )

            target_tensors_gender_man = subtask_face_gender_result[0, :][
                sub_peak_inds.transpose(0, 1).tolist()
            ].view(-1, 1)
            target_tensors_gender_woman = subtask_face_gender_result[1, :][
                sub_peak_inds.transpose(0, 1).tolist()
            ].view(-1, 1)

            target_tensors_age = subtask_face_age_result[0, :][
                sub_peak_inds.transpose(0, 1).tolist()
            ].view(-1, 1)
            target_tensors_headpose_pitch = subtask_face_headpose_result[0, :][
                sub_peak_inds.transpose(0, 1).tolist()
            ].view(-1, 1)
            target_tensors_headpose_yaw = subtask_face_headpose_result[1, :][
                sub_peak_inds.transpose(0, 1).tolist()
            ].view(-1, 1)

            gender_scores_man = target_tensors_gender_man.squeeze(0).cpu().sigmoid()
            gender_scores_woman = target_tensors_gender_woman.squeeze(0).cpu().sigmoid()

            if not self.tflite:
                headpose_scores_pitch = target_tensors_headpose_pitch.squeeze(0).cpu()
                headpose_scores_yaw = target_tensors_headpose_yaw.squeeze(0).cpu()
            else:
                headpose_scores_pitch = (
                    target_tensors_headpose_pitch.squeeze(0).cpu().sigmoid()
                )
                headpose_scores_yaw = (
                    target_tensors_headpose_yaw.squeeze(0).cpu().sigmoid()
                )

            age_attribute = target_tensors_age.squeeze(0).cpu()

            peak_inds = torch.cat((peak_inds, peak_inds), dim=1)[
                :, 1:3
            ]  # trans position to coordinates, trans from column and row to x and y
            input_centers = (peak_inds + offsets) * feat_stride

            sub_peak_inds = torch.cat((sub_peak_inds, sub_peak_inds), dim=1)[:, 1:3]
            input_centers_subtask = (sub_peak_inds + sub_task_offsets) * feat_stride

            bboxes = torch.cat(
                (
                    (input_centers[:, 0] - sizes[:, 0] / 2).unsqueeze(1),
                    (input_centers[:, 1] - sizes[:, 1] / 2).unsqueeze(1),
                    (input_centers[:, 0] + sizes[:, 0] / 2).unsqueeze(1),
                    (input_centers[:, 1] + sizes[:, 1] / 2).unsqueeze(1),
                ),
                dim=1,
            ).type_as(det_body_bboxes)

            sub_bboxes = torch.cat(
                (
                    (input_centers_subtask[:, 0] - sub_task_sizes[:, 0] / 2).unsqueeze(
                        1
                    ),
                    (input_centers_subtask[:, 1] - sub_task_sizes[:, 1] / 2).unsqueeze(
                        1
                    ),
                    (input_centers_subtask[:, 0] + sub_task_sizes[:, 0] / 2).unsqueeze(
                        1
                    ),
                    (input_centers_subtask[:, 1] + sub_task_sizes[:, 1] / 2).unsqueeze(
                        1
                    ),
                    sub_scores,
                ),
                dim=1,
            ).type_as(det_face_bboxes)

            if center_name == "main_task_center":
                sub_task_center_offsets = torch.cat(
                    (
                        subtask_center_offset_result[0, :][is_peak].view(-1, 1),
                        subtask_center_offset_result[1, :][is_peak].view(-1, 1),
                    ),
                    dim=1,
                )
                sub_task_offset = torch.cat(
                    (
                        (input_centers[:, 0]).unsqueeze(1),
                        (input_centers[:, 1]).unsqueeze(1),
                        (
                            input_centers[:, 0]
                            + sub_task_center_offsets[:, 0] * feat_stride
                        ).unsqueeze(1),
                        (
                            input_centers[:, 1]
                            + sub_task_center_offsets[:, 1] * feat_stride
                        ).unsqueeze(1),
                    ),
                    dim=1,
                ).type_as(det_body_bboxes)

            else:
                print("error")

            inds = nms(
                boxes=bboxes,
                scores=scores.view(-1).to("cpu"),
                iou_threshold=cfg["nms_thr"],
            )
            bboxes = bboxes[inds]

            sub_bboxes = sub_bboxes[inds]

            genders_man = gender_scores_man[inds]
            genders_woman = gender_scores_woman[inds]
            genders = torch.zeros([genders_man.shape[0], 2])

            headpose_pitch = headpose_scores_pitch[inds]
            headpose_yaw = headpose_scores_yaw[inds]
            headposes = torch.zeros([headpose_pitch.shape[0], 2])

            for i in range(genders_man.shape[0]):
                genders[i, 0] = genders_man[i]
                genders[i, 1] = genders_woman[i]
            for i in range(headpose_pitch.shape[0]):
                headposes[i, 0] = (headpose_pitch[i] - 0.5) * 180
                headposes[i, 1] = (headpose_yaw[i] - 0.5) * 360

            ages = age_attribute[inds]

            if center_name == "main_task_center":
                sub_task_offset_1 = sub_task_offset[inds]

            if ages.shape == torch.Size([1]):
                ages = ages.unsqueeze(0)

            det_body_bboxes = torch.cat((det_body_bboxes, bboxes), dim=0)
            det_face_bboxes = torch.cat((det_face_bboxes, sub_bboxes), dim=0)
            det_center_offset = torch.cat((det_center_offset, sub_task_offset_1), dim=0)
            det_face_attributes = torch.cat(
                (genders, ages, headposes), dim=1
            )  # add binoeye gaze
            extras_tensors = [
                subtask_cur_map.type_as(det_center_offset),
                det_center_offset,
            ]

        if rescale and det_body_bboxes.numel() > 0:
            det_body_bboxes[:, :4] /= img_meta["scale_factor"]
            det_face_bboxes[:, :4] /= img_meta["scale_factor"]

        return (
            det_body_bboxes,
            det_face_bboxes,
            det_face_attributes,
            extras_tensors,
        )
