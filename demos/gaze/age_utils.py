# age_utils.py
import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms


def get_age_range(age):
    """
    根据年龄返回对应的年龄范围文本
    """
    if age < 15:
        return "15以下"
    elif age < 25:
        return "15-25"
    elif age < 35:
        return "25-35"
    elif age < 45:
        return "35-45"
    elif age < 55:
        return "45-55"
    elif age < 65:
        return "55-65"
    else:
        return "65以上"


def is_frontal_face(pitch, yaw, roll, pitch_thres=20, yaw_thres=20, roll_thres=90):
    """
    Determine if a face is frontal based on pitch, yaw, and roll angles.

    Args:
        pitch, yaw, roll: Head pose angles in degrees
        threshold: Maximum allowed angle deviation in degrees

    Returns:
        Boolean indicating if the face is frontal
    """
    return abs(pitch) < pitch_thres and abs(yaw) < yaw_thres and abs(roll) < roll_thres


def update_person_age_gender(
    tid,
    age,
    gender,
    age_predictions,
    gender_predictions,
    final_age_gender,
    min_samples=4,
    max_samples=5,
):
    """
    更新指定目标的年龄和性别预测数据，并计算最终结果。

    Args:
        tid: 目标 ID
        age: 当前帧预测的年龄
        gender: 当前帧预测的性别
        age_predictions: 存储各目标年龄预测的字典
        gender_predictions: 存储各目标性别预测的字典
        final_age_gender: 存储各目标最终年龄和性别的字典
        min_samples: 计算最终结果所需最少样本数
        max_samples: 保存的最大样本数

    Returns:
        (final_age, final_gender) 如果样本足够，否则返回 (None, None)
    """
    if tid not in age_predictions:
        age_predictions[tid] = []
        gender_predictions[tid] = []
    age_predictions[tid].append(float(age))
    gender_predictions[tid].append(gender)

    # 限制最大存储样本数
    if len(age_predictions[tid]) > max_samples:
        age_predictions[tid].pop(0)
        gender_predictions[tid].pop(0)

    if len(age_predictions[tid]) >= min_samples:
        final_age = int(sum(age_predictions[tid]) / len(age_predictions[tid]))
        # 多数投票决定性别
        final_gender = max(
            set(gender_predictions[tid]), key=gender_predictions[tid].count
        )
        final_age_gender[tid] = (final_age, final_gender)
        return final_age, final_gender
    return None, None
