# -*- coding: utf8 -*-
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
from utils import int2str

from constants import TO_DEL_LINES, NUM_QUESTIONS, NUM_TRAIN_DEPRESSION, NUM_TRAIN_SUBJECTS, NUM_FEATURES, \
    SELECTED_INDICES


def read_feature(feature_dir, subject_no, question_no=None, selected_indices=None):
    """
    读取opensmile提取的特征文件(名为audio_feature<no>.txt的文件，包含了单个受试者的所有问题的特征)。
    :param feature_dir: 音频特征文件所在目录
    :param subject_no: 受试者编号，从1开始
    :param question_no: 问题编号，从1开始。
    :param selected_indices: 选择的特征序号，从0开始。例如：[0, 52, 62, 74, 333, 1581]
    :return: 如果指定了question_no，则返回该问题的特征向量；否则返回所有问题的特征向量
    """
    feature_filename = f'audio_feature{int2str(subject_no)}.txt'
    txt_path = os.path.join(feature_dir, feature_filename)
    assert os.path.exists(txt_path)
    with open(txt_path) as f:
        vectors = f.readlines()
    if question_no is not None:
        vector = vectors[TO_DEL_LINES + question_no - 1].split(',')
        vector = np.array(vector)
        vector = vector[1:NUM_FEATURES + 1]
        if selected_indices is not None:
            vector = vector[selected_indices]
        return vector
    else:
        vectors = np.array(
            [vector.split(',') for vector in vectors[TO_DEL_LINES + 1:TO_DEL_LINES + NUM_QUESTIONS + 1]])
        vectors = vectors[:, 1:NUM_FEATURES + 1]
        if selected_indices is not None:
            vectors = vectors[:, selected_indices]
        return vectors


def process_data(feature_dir, split=True, to_tensor=True):
    """
    获取用于训练模型的数据
    :param feature_dir: 特征所在路径
    :param split: 是否分割数据集
    :param to_tensor: 是否转换为torch张量。默认为True，转换为张量。设置为False时，返回的是numpy数组。
    :return: 返回(train_features, val_features, train_labels, val_labels)。
    在不分割数据集的情况下，val_features和val-labels为None。
    返回的特征形状为：(NUM_QUESTIONS, NUM_SUBJECTS, DIM_FEATURES)
    """
    # NOTE:预计获取到的特征形状为：(NUM_QUESTIONS, NUM_SUBJECTS, DIM_FEATURES)

    # 获取标签。每个问题的标签都是相同的。
    labels = np.array([1 if i <= NUM_TRAIN_DEPRESSION else 0 for i in range(1, NUM_TRAIN_SUBJECTS + 1)])

    # 获取特征
    train_features = []
    val_features = [] if split else None
    train_labels = []
    val_labels = [] if split else None

    for question_no in range(1, NUM_QUESTIONS + 1):
        feature_dir = os.path.join(feature_dir, 'train_enhance')
        train_data = np.array(
            [read_feature(feature_dir, subject_no, question_no=question_no, selected_indices=SELECTED_INDICES)
             for subject_no in range(1, NUM_TRAIN_SUBJECTS + 1)]
        )

        # 划分训练集和验证集
        if split:
            # 分割比例在这里写死了，shuffle的随机种子也写死了
            x_train, x_val, y_train, y_val = train_test_split(train_data, labels, test_size=0.2, random_state=42)
            train_features.append(x_train)
            val_features.append(x_val)
            train_labels.append(y_train)
            val_labels.append(y_val)
        else:
            train_features.append(train_data)
            train_labels.append(labels)

    # 转换为numpy数组
    train_features = np.array(train_features)
    val_features = np.array(val_features)
    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)

    # 转换为张量
    if to_tensor:
        train_features = torch.tensor(train_features)
        val_features = torch.tensor(val_features)
        train_labels = torch.tensor(train_labels)
        val_labels = torch.tensor(val_labels)

    # 返回
    return train_features, val_features, train_labels, val_labels


def get_dataloader(features, labels, batch_size=1, shuffle=True):
    """
    获取数据集的DataLoader
    :param features: 特征张量
    :param labels: 标签张量
    :param batch_size: 批量大小，默认为1
    :param shuffle: 是否打乱，默认为True
    :return: 将特征和标签构造为TensorDataset后生成的DataLoader
    """
    dataset = TensorDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
