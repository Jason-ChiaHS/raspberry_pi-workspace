import base64
import concurrent
import concurrent.futures
import multiprocessing
import queue
import threading
import time
import typing as t
from cProfile import Profile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
import torch.nn as nn
from picamera2 import CompletedRequest, MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.request import Helpers
from torchvision.ops import nms
from urllib3.exceptions import MaxRetryError

from sdk.cam import Cam
from sdk.helpers.config import ConfigLoader

CLIENT_SERVER_URL = "http://10.68.254.109:8020"
CONFIG_PATH = "./config.toml"

# Picamera2.set_logging(Picamera2.DEBUG)


class Cam:
    # take in params from argparse
    def __init__(
        self,
        model_path: Path,
        debug: bool,
        profiling: bool,
        width: int,
        height: int,
    ):
        self.model_path = model_path
        self.debug = debug
        self.profiling = profiling
        self.width = width
        self.height = height

        self.imx500 = IMX500(self.model_path)
        self.imx500.show_network_fw_progress_bar()

        (self.input_width, self.input_height) = self.imx500.get_input_size()

        self.picam2 = Picamera2(self.imx500.camera_num)
        print(self.picam2.sensor_modes)
        self.config = self.picam2.create_video_configuration(
            {"size": (self.width, self.height)},
            # buffer_count=1,
            # buffer_count=10,
        )
        self.picam2.configure(self.config)
        print(self.picam2.camera_configuration()["raw"])

        self.picam2.pre_callback = self.camera_callback

    def start(self):
        self.picam2.start()

    def camera_callback(self, request: CompletedRequest):
        pass


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


def post_processing_tensor(
    np_outputs,
    tflite=False,
):
    """
    runs the post processing on the output tensor from the metadata of the request
    """

    cfg = {"kernel_size": 3, "score_thr": 0.3, "nms_thr": 0.3}

    # Not used, will rescale using another function later
    img_meta = {"scale_factor": 1}
    rescale = False
    feat_stride = 16
    center_name = "main_task_center"

    if not tflite:
        # using onnx model
        # C H W, 0 1 2
        center_map = squeeze_tensor(np_outputs[0], squeezeN=True)  # output
        offset_result = squeeze_tensor(np_outputs[1], squeezeN=True)  # 825
        size_result = squeeze_tensor(np_outputs[2], squeezeN=True)  # 853
        subtask_box_center_result = squeeze_tensor(np_outputs[6], squeezeN=True)  # 890
        subtask_box_offset_result = squeeze_tensor(np_outputs[7], squeezeN=True)  # 892
        subtask_box_size_result = squeeze_tensor(np_outputs[8], squeezeN=True)  # 902
        subtask_center_offset_result = squeeze_tensor(
            np_outputs[3], squeezeN=True
        )  # 829
        subtask_face_gender_result = squeeze_tensor(np_outputs[5], squeezeN=True)  # 833
        # Could be empty??
        subtask_face_age_result = squeeze_tensor(np_outputs[9], squeezeN=True)  # 904
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
        subtask_face_gender_result = squeeze_tensor(np_outputs[4], squeezeN=True)  # 833
        subtask_face_gender_result = tt_tensor(subtask_face_gender_result)
        subtask_face_headpose_result = squeeze_tensor(
            np_outputs[10], squeezeN=True
        )  # 900
        subtask_face_headpose_result = tt_tensor(subtask_face_headpose_result)
        subtask_box_center_result = squeeze_tensor(np_outputs[6], squeezeN=True)  # 890
        subtask_box_center_result = tt_tensor(subtask_box_center_result)
        subtask_box_offset_result = squeeze_tensor(np_outputs[7], squeezeN=True)  # 892
        subtask_box_offset_result = tt_tensor(subtask_box_offset_result)
        subtask_box_size_result = squeeze_tensor(np_outputs[8], squeezeN=True)  # 902
        subtask_box_size_result = tt_tensor(subtask_box_size_result)
        # Could be empty??
        subtask_face_age_result = squeeze_tensor(np_outputs[9], squeezeN=True)  # 904
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
        sub_scores = subtask_cur_map[sub_peak_inds.transpose(0, 1).tolist()].view(-1, 1)
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

        headpose_scores_pitch = target_tensors_headpose_pitch.squeeze(0).cpu().sigmoid()
        headpose_scores_yaw = target_tensors_headpose_yaw.squeeze(0).cpu().sigmoid()

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
                (input_centers_subtask[:, 0] - sub_task_sizes[:, 0] / 2).unsqueeze(1),
                (input_centers_subtask[:, 1] - sub_task_sizes[:, 1] / 2).unsqueeze(1),
                (input_centers_subtask[:, 0] + sub_task_sizes[:, 0] / 2).unsqueeze(1),
                (input_centers_subtask[:, 1] + sub_task_sizes[:, 1] / 2).unsqueeze(1),
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
        # extras_tensors,
    )


def make_array(buffer, config):
    """Make a 2d numpy array from the named stream's buffer."""
    array = buffer
    fmt = config["format"]
    w, h = config["size"]
    stride = config["stride"]

    # Turning the 1d array into a 2d image-like array only works if the
    # image stride (which is in bytes) is a whole number of pixels. Even
    # then, if they don't match exactly you will get "padding" down the RHS.
    # Working around this requires another expensive copy of all the data.
    if fmt in ("BGR888", "RGB888"):
        if stride != w * 3:
            array = array.reshape((h, stride))
            array = np.asarray(array[:, : w * 3], order="C")
        image = array.reshape((h, w, 3))
    elif fmt in ("XBGR8888", "XRGB8888"):
        if stride != w * 4:
            array = array.reshape((h, stride))
            array = np.asarray(array[:, : w * 4], order="C")
        image = array.reshape((h, w, 4))
    elif fmt in ("BGR161616", "RGB161616"):
        if stride != w * 6:
            array = array.reshape((h, stride))
            array = np.asarray(array[:, : w * 6], order="C")
        array = array.view(np.uint16)
        image = array.reshape((h, w, 3))
    elif fmt in ("YUV420", "YVU420"):
        # Returning YUV420 as an image of 50% greater height (the extra bit continaing
        # the U/V data) is useful because OpenCV can convert it to RGB for us quite
        # efficiently. We leave any packing in there, however, as it would be easier
        # to remove that after conversion to RGB (if that's what the caller does).
        image = array.reshape((h * 3 // 2, stride))
    elif fmt in ("YUYV", "YVYU", "UYVY", "VYUY"):
        # These dimensions seem a bit strange, but mean that
        # cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV) will convert directly to RGB.
        image = array.reshape(h, stride // 2, 2)
    else:
        raise RuntimeError("Format " + fmt + " not supported")
    return image


def post_processing(np_outputs, timestamp):
    # print([n.shape for n in np_outputs])
    output = post_processing_tensor(np_outputs, True)
    # print([type(o) for o in output])
    output = [o.tolist() for o in output]
    # image = img
    # image = cv2.resize(image, (1920, 1080))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # _, encoded_image = cv2.imencode(".jpg", image)
    # frame = encoded_image.tobytes()

    filename = datetime.now().timestamp()
    filename = timestamp

    image_url = f"{CLIENT_SERVER_URL}/image/{filename}"
    meta_url = f"{CLIENT_SERVER_URL}/meta/{filename}"

    metadata = {
        "bboxes": output[0],
        "face_bboxes": output[1],
        "attr": output[2],
        # "extra": output[3],
    }

    # filename, img, metadata
    # print("here?")
    return (filename, metadata)

    # requests.put(image_url, data="", timeout=3)
    # try:
    #     requests.put(
    #         meta_url,
    #         json={
    #             "bboxes": output[0],
    #             "face_bboxes": output[1],
    #             "attr": output[2],
    #             "extra": output[3],
    #         },
    #         headers={"Content-Type": "application/json"},
    #         timeout=3,
    #     )
    # except:
    #     pass


def thread_send_data(jobs: queue.Queue):
    while True:
        try:
            job = jobs.get(timeout=4)
        except:
            continue
        print(f"qsize: {jobs.qsize()}")
        # while jobs.qsize() > 100:
        #     jobs.get()

        img, np_outputs, filename = job[0], job[1], job[2]
        # job = t.cast(concurrent.futures.Future, job)
        # filename, metadata = j.result()
        output = post_processing_tensor(np_outputs, True)
        output = [o.tolist() for o in output]
        metadata = {
            "bboxes": output[0],
            "face_bboxes": output[1],
            "attr": output[2],
            # "extra": output[3],
        }

        image = img
        image = cv2.resize(image, (1920, 1080))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _, encoded_image = cv2.imencode(".jpg", image)
        frame = encoded_image.tobytes()

        image_url = f"{CLIENT_SERVER_URL}/image/{filename}"
        meta_url = f"{CLIENT_SERVER_URL}/meta/{filename}"
        try:
            requests.put(image_url, data=frame, timeout=3)
            requests.put(
                meta_url,
                json=metadata,
                headers={"Content-Type": "application/json"},
                timeout=3,
            )
        except Exception as e:
            pass


def apply_timestamp(request: CompletedRequest):
    colour = (0, 255, 0)
    origin = (0, 30)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    thickness = 2
    timestamp = str(datetime.now().timestamp())
    request.timestamp = timestamp
    with MappedArray(request, "main") as m:
        cv2.putText(m.array, timestamp, origin, font, scale, colour, thickness)


if __name__ == "__main__":
    config = ConfigLoader(CONFIG_PATH)
    cam = Cam(
        config.model_path,
        config.profiling,
        config.width,
        config.height,
    )
    cam.start()
    cam.picam2.pre_callback = apply_timestamp
    time.sleep(2)

    # jobs = queue.Queue()
    # jobs = multiprocessing.Queue()
    jobs = queue.SimpleQueue()
    i = 0
    m_count = 0
    # thread = threading.Thread(target=thread_send_data, args=(jobs,), daemon=True)
    # thread.start()

    # Number of worker threads
    num_threads = 4

    # Start worker threads using ThreadPoolExecutor
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)

    # Submit worker threads that will continuously fetch from the queue
    for i in range(num_threads):
        executor.submit(thread_send_data, jobs)

    try:
        with Profile() as pr:
            # with concurrent.futures.ThreadPoolExecutor() as executor:
            # with multiprocessing.Pool(processes=4) as pool:
            with concurrent.futures.ProcessPoolExecutor(max_workers=3) as p_executor:
                start = time.monotonic()
                # while i < 50:
                while True:
                    # while m_count < 500:
                    # cam.picam2.capture_array("main")

                    request = t.cast(CompletedRequest, cam.picam2.capture_request())
                    request.timestamp = str(datetime.now().timestamp())
                    metadata = request.get_metadata()

                    # img = request.make_buffer("main")
                    # img = make_array(img, request.config["main"])

                    if metadata is not None:
                        np_outputs = cam.imx500.get_outputs(metadata)
                        if np_outputs is not None:
                            m_count += 1
                            print(m_count)
                            img = request.make_array("main").copy()
                            # img = request.make_buffer("main")
                            # img = 0

                            # async_result = executor.submit(
                            #     post_processing,
                            #     np_outputs,
                            #     img,
                            # )
                            # async_result.result()

                            # async_result = pool.apply_async(
                            #     post_processing,
                            #     (np_outputs, img),
                            # )
                            # async_result.get()
                            # print(f"got new inference, {request.timestamp}")
                            jobs.put((img, np_outputs, request.timestamp))
                            # print(f"after, {request.timestamp}")
                    request.release()
                    # print("release")

                    # print(request.ref_count)
                    i += 1

            # for _ in range(num_threads):
            #     jobs.put(None)

            # Calculate fps
            end = time.monotonic()
            print(f"Camera FPS: {i / (end - start)}")
            print(f"Model FPS: {m_count / (end - start)}")
            print(f"i: {i}")
            print(f"m_count: {m_count}")
            cam.picam2.stop()
            pr.dump_stats(Path(__file__).parent / "pp.prof")
    except KeyboardInterrupt:
        pass
