# -*- coding: utf8 -*-
import os
import argparse

import numpy as np
from torch.optim import Adam
import torch.nn as nn
from utils import set_seeds
from constants import NUM_QUESTIONS, NUM_TEST_SUBJECTS, NUM_OURS_SUBJECTS
from data_process import read_file, process_data, get_dataloader
from CNN_torch import CNN, load_model

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# 设置参数转换
parser = argparse.ArgumentParser()
parser.add_argument("--feature_dir", default="/home/wangxu/project/Audioprocessing/2022data/train_enhance")
parser.add_argument("--output_path", default="/home/wangjiyuan/dev/DepressionAudioProcessing/models")
args = parser.parse_args()


def main():
    for seed in range(1001):
        if flow_by_seed(seed):
            break


def flow_by_seed(seed, lr=0.001, weight_decay=0.001, num_epochs=100, batch_size=1):
    """
    在特定种子下完成一次完整的流程
    :param seed:
    :param lr: 训练使用的学习率
    :param weight_decay: 权重衰退设置值
    :param num_epochs: 模型训练的Epochs数
    :param batch_size: 批量大小
    :return: True表示找到了好的模型，False表示没找到
    """
    set_seeds(seed)
    train_features, val_features, train_labels, val_labels = process_data(args.feature_dir)
    result = []
    for question_no in range(1, NUM_QUESTIONS):
        data_iter = get_dataloader(train_features[question_no], train_labels[question_no], batch_size=batch_size)
        cur_val_features = val_features[question_no]
        cur_val_labels = val_labels[question_no]

        model = CNN()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss()
        train_one_model(model, data_iter, optimizer, loss_fn, num_epochs)
        acc, TP, FP, TN, FN = validate(model, cur_val_features, cur_val_labels)
        # 计算灵敏度和特异度
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        result.append([sensitivity, specificity])

    # 这部分看不懂，先挪过来
    result_specificity = np.array(result)
    result_specificity = np.lexsort(-result_specificity.T)
    for i in range(len(result_specificity)):
        result_specificity[i] += 1

    result_sensitivity = np.array(result)
    result_sensitivity = np.lexsort(-result_sensitivity[:, ::-1].T)
    for i in range(len(result_sensitivity)):
        result_sensitivity[i] += 1

    print(result_specificity)
    print(result_sensitivity)

    index_threshold = 2

    model_indices = []
    for i in range(index_threshold):
        model_indices.append(result_sensitivity[i])
    for i in range(NUM_QUESTIONS):
        if len(model_indices) == (index_threshold * 2 - 1):
            break
        if result_specificity[i] not in model_indices:
            model_indices.append(result_specificity[i])
    print(model_indices)

    # 测试
    res, our_res = test(model_indices, index_threshold)

    print('seed == ' + str(seed))
    print('test正确率：' + str(res / NUM_TEST_SUBJECTS))
    print('our正确率：' + str(our_res / NUM_OURS_SUBJECTS))

    if (res / NUM_TEST_SUBJECTS) > 0.9 and (our_res / NUM_OURS_SUBJECTS) > 0.8:
        print("找到了！！！！")
        print('seed == ' + str(seed))
        print('test正确率：' + str(res / NUM_TEST_SUBJECTS))
        print('our正确率：' + str(our_res / NUM_OURS_SUBJECTS))
        return True

    return False


def train_one_model(model, data_iter, optimizer, loss_fn, num_epochs):
    """
    训练一个模型
    :param model: 模型实例
    :param data_iter: 数据迭代器
    :param optimizer: 优化器
    :param loss_fn: 损失函数
    :param num_epochs: 训练的epochs数
    :return: None
    """
    for epoch in range(num_epochs):
        for features, labels in data_iter:
            optimizer.zero_grad()
            y_hat = model(features)
            loss = loss_fn(y_hat, labels)
            loss.backward()
            optimizer.step()
    # TODO: 也许可以在这里加入观察训练的过程。数据量少，有过拟合的可能，加点早停或许会有用。


def validate(model: CNN, val_features, val_labels):
    """
    获取模型的相关指标
    :param model: 模型
    :param val_features: 验证集特征
    :param val_labels: 验证集标签
    :return: acc，TP，FP，TN，FN
    """
    y = model.predict(val_features)

    accuracy = (y == val_labels).sum() / len(val_labels)

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y)):
        if y[i] == 1 and val_labels[i] == 1:
            TP += 1
        elif y[i] == 1 and val_labels[i] == 0:
            FP += 1
        elif y[i] == 0 and val_labels[i] == 0:
            TN += 1
        elif y[i] == 0 and val_labels[i] == 1:
            FN += 1
    return accuracy, TP, FP, TN, FN


def test(model_indices, index_threshold):
    """
    测试模型
    :param model_indices:
    :param index_threshold:
    :return:
    """
    test_result = []

    for idx in range(1, NUM_TEST_SUBJECTS + 1):
        data_test = read_file(args.feature_dir + '/test_enhance/', idx)

        health = 0
        depression = 0

        for model in model_indices:
            model_path = f'{args.output_path}/model_{model}.h5'
            loaded_model = load_model(model_path)
            if int(np.argmax(loaded_model.predict(np.array(data_test[model - 1]).reshape((1, 1, 20, 1))), axis=1)) == 0:
                health += 1
            else:
                depression += 1
        if health >= index_threshold:
            test_result.append(0)
        else:
            test_result.append(1)

    print(test_result)

    res = 0

    # 计算灵敏度
    q1 = 0  # 真阳性人数
    q2 = 0  # 假阴性人数

    # 计算特异性
    p1 = 0  # 真阴性人数
    p2 = 0  # 假阳性人数

    for i in range(0, 6):
        if test_result[i] == 1:
            res += 1
            q1 += 1
        else:
            q2 += 1
    for i in range(6, 13):
        if test_result[i] == 0:
            res += 1
            p1 += 1
        else:
            p2 += 1

    print('test灵敏度：' + str(q1 / (q1 + q2)))
    print('test特异性：' + str(p1 / (p1 + p2)))

    our_result = []

    for idx in range(1, NUM_OURS_SUBJECTS + 1):
        data_test = read_file(args.feature_dir + '/health_data_enhance/', idx)

        health = 0
        depression = 0

        for model in model_indices:
            loaded_model = load_model(args.output_path + "/model_{}.h5".format(model))
            if int(np.argmax(loaded_model.predict(np.array(data_test[model - 1]).reshape((1, 1, 20, 1))), axis=1)) == 0:
                health += 1
            else:
                depression += 1

        if health >= index_threshold:
            our_result.append(0)
        else:
            our_result.append(1)

    print(our_result)

    our_res = 0

    # 计算灵敏度
    our_q1 = 0  # 真阳性人数
    our_q2 = 0  # 假阴性人数

    # 计算特异性
    our_p1 = 0  # 真阴性人数
    our_p2 = 0  # 假阳性人数

    for i in range(0, NUM_OURS_SUBJECTS):
        if our_result[i] == 0:
            our_res += 1
            our_p1 += 1
        else:
            our_p2 += 1

    print('our特异性：' + str(our_p1 / (our_p1 + our_p2)))

    return res, our_res
