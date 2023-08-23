# -*- coding: utf8 -*-
import os
import numpy as np

# opensmile提取的文件，前1589行是特征说明，1590行开始是特征信息
TO_DEL_LINES = 1589
NUM_QUESTION = 29
# 选择的特征序号
b = [1492, 227, 1403, 174, 468, 1493, 1036, 417, 607, 612, 45, 15, 1277, 872, 153, 1491, 619, 685, 305, 529]


def read_file(path, i):
    """
    读取opensmile提取的特征文件
    :param path:特征路径
    :param i: 受试者编号
    :return: 特征序列
    """
    feature_name = f'audio_feature{str(subject_no).rjust(3, "0")}.txt'
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
    for x in range(0, NUM_QUESTION):
        result = [float(matrix[x][c]) for c in b]
        # features_mat = mat[x]
        # features_line = [float(features_mat[j]) for j in range(1280)]
        # result = result + features_line
        a.append(result)
    arr = np.array(a)
    f.close()
    return arr


def read_file_index(index, path, subject_no):
    """
    读取opensmile提取的特征文件
    :param index: 问题序号，从1开始
    :param path: 特征路径
    :param subject_no: 受试者序号
    :return: 特征
    """
    feature_name = f'audio_feature{str(subject_no).rjust(3, "0")}.txt'
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
    result = [float(matrix[index - 1][c]) for c in b]

    # mat = hdf5.loadmat(mat_path)['dataset_features'][index-1]
    # features_mat = [float(mat[i]) for i in range(1280)]
    # result = result + features_mat

    f.close()
    return result


def process_data():
    train_data = read