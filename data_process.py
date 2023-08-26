# -*- coding: utf8 -*-
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split

from constants import TO_DEL_LINES, NUM_QUESTIONS, NUM_TRAIN_DEPRESSION, NUM_TRAIN_SUBJECTS

b = [1492, 227, 1403, 174, 468, 1493, 1036, 417, 607, 612, 45, 15, 1277, 872, 153, 1491, 619, 685, 305, 529]
"""
选择的特征序号
"""


def read_file(path, subject_no):
    """
    读取opensmile提取的特征文件
    :param path:特征路径
    :param subject_no: 受试者编号
    :return: 特征序列
    """
    feature_name = f'audio_feature0{subject_no}.txt'
    txt_path = os.path.join(path, feature_name)
    # mat_path = path + 'whisper_features0{}.mat'.format(i)
    f = open(txt_path)
    lines = f.readlines()

    del lines[0:TO_DEL_LINES]
    first_ele = True
    for data in lines:
        data = data.strip('\n')
        nums = data.split(',')
        if first_ele:
            matrix = np.array(nums)
            first_ele = False
        else:
            matrix = np.c_[matrix, nums]
    matrix = matrix.transpose()

    # mat = hdf5.loadmat(mat_path)['dataset_features']
    a = []
    for x in range(0, NUM_QUESTIONS):
        result = [float(matrix[x][c]) for c in b]
        # features_mat = mat[x]
        # features_line = [float(features_mat[j]) for j in range(1280)]
        # result = result + features_line
        a.append(result)
    arr = np.array(a)
    f.close()
    return arr


def read_file_index(question_no, path, subject_no):
    """
    读取opensmile提取的特征文件
    :param question_no: 问题序号，从1开始
    :param path: 特征路径
    :param subject_no: 受试者序号
    :return: 特征
    """
    feature_name = f'audio_feature0{subject_no}.txt'
    txt_path = os.path.join(path, feature_name)
    # mat_path = path + 'whisper_features0{}.mat'.format(i)
    f = open(txt_path)
    lines = f.readlines()
    del lines[0:TO_DEL_LINES]
    first_ele = True
    for data in lines:
        data = data.strip('\n')
        nums = data.split(',')
        if first_ele:
            matrix = np.array(nums)
            first_ele = False
        else:
            matrix = np.c_[matrix, nums]
    matrix = matrix.transpose()
    result = [float(matrix[question_no - 1][c]) for c in b]

    # mat = hdf5.loadmat(mat_path)['dataset_features'][index-1]
    # features_mat = [float(mat[i]) for i in range(1280)]
    # result = result + features_mat

    f.close()
    return result


def process_data(feature_dir, split=True, to_tensor=True):
    """
    获取用于训练模型的数据
    :param feature_dir: 特征所在路径
    :param split: 是否分割数据集
    :param to_tensor: 是否转换为torch张量
    :return: 返回(train_features, val_features, train_labels, val_labels)。
    在不分割数据集的情况下，val_features和val-labels为None。
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
        feature_path = os.path.join(feature_dir, 'train_enhance')
        train_data = np.array(
            [read_file_index(question_no, feature_path, subject_no) for subject_no in range(1, NUM_TRAIN_SUBJECTS + 1)]
        )

        # 划分训练集和验证集
        if split:
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
