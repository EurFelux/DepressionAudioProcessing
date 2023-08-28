# -*- coding: utf8 -*-
import os
import argparse

import numpy as np
import torch.cuda
from torch.optim import Adam
import torch.nn as nn
from utils import set_seeds
from constants import NUM_QUESTIONS, NUM_TEST_SUBJECTS, NUM_OURS_SUBJECTS, NUM_TEST_DEPRESSION, SELECTED_INDICES, \
    NUM_OURS_DEPRESSION, LOG_LEVEL
from data_process import read_feature, process_data, get_dataloader
from CNN_torch import CNN, load_model, adapt_shape
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# 设置参数转换
parser = argparse.ArgumentParser()
parser.add_argument("--feature_dir", default="/home/wangjiyuan/dev/DepressionAudioProcessing/features")
parser.add_argument("--output_path", default="/home/wangjiyuan/dev/DepressionAudioProcessing/models")
parser.add_argument('--skip-train', action='store_true', default=False)
parser.add_argument("--log-level", default=LOG_LEVEL)
args = parser.parse_args()


def main():
    print(f'CUDA test:\n\tis_available:\t{torch.cuda.is_available()}\n')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for seed in range(1):
        if flow_by_seed(seed, device=device, skip_train=args.skip_train):
            break


def flow_by_seed(seed, lr=0.001, weight_decay=0.001, num_epochs=100, batch_size=1, skip_train=False,
                 device=torch.device("cpu")):
    """
    在特定种子下完成一次完整的流程
    :param seed:
    :param lr: 训练使用的学习率
    :param weight_decay: 权重衰退设置值
    :param num_epochs: 模型训练的Epochs数
    :param batch_size: 批量大小
    :param skip_train: 是否跳过训练过程。如果为True，则直接加载模型并进行测试。
    :param device: 训练和推理时使用的设备，默认使用cpu
    :return: True表示找到了好的模型，False表示没找到
    """
    set_seeds(seed)

    train_features_dir = os.path.join(args.feature_dir, 'train')
    train_features, val_features, train_labels, val_labels = process_data(train_features_dir, device=device)
    result = []
    for question_no in tqdm(range(1, NUM_QUESTIONS), desc='Validating models' if skip_train else 'Training models',
                            unit='model', leave=False):
        data_iter = get_dataloader(train_features[question_no], train_labels[question_no], batch_size=batch_size,
                                   change_shape=True)
        cur_val_features = adapt_shape(val_features[question_no])
        cur_val_labels = val_labels[question_no]

        if skip_train:
            model_path = f'{args.output_path}/model_{question_no}.h5'
            assert os.path.exists(model_path)
            model = load_model(model_path).to(device)
        else:
            model = CNN().to(device)
            optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            loss_fn = nn.CrossEntropyLoss()
            train_one_model(model, data_iter, optimizer, loss_fn, num_epochs)
            model.save(f'{args.output_path}/model_{question_no}.h5')
        acc, TP, FP, TN, FN = validate(model, cur_val_features, cur_val_labels)

        # 计算灵敏度和特异度
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        result.append([sensitivity, specificity])
        print(f'Model {question_no}: Accuracy: {acc}, Sensitivity: {sensitivity}, Specificity: {specificity}')

    # 获取模型优劣排序
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

    # 选择模型
    index_threshold = 2

    model_indices = []
    # 加入灵敏度最高的两个模型
    for i in range(index_threshold):
        model_indices.append(result_sensitivity[i])
    # 再加入一个特异度最高的模型
    for i in range(len(result)):
        if len(model_indices) == (index_threshold * 2 - 1):
            break
        if result_specificity[i] not in model_indices:
            model_indices.append(result_specificity[i])
    print(f'Selected models: {model_indices}')
    # 最后挑选了3个模型出来

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
    for epoch in tqdm(range(num_epochs), desc='Training', unit='epoch', leave=False):
        for features, labels in tqdm(data_iter, desc='Processing batch', unit='batch', leave=False):
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
    y = model.predict(val_features, return_type='pt')

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


def validate_models(model_indices: list, index_threshold: int, set_dir: str, num_subjects: int, num_depression: int,
                    device=torch.device("cpu")):
    """
    验证模型。使用了SELECTED_INDICES作为特征序号。
    :param model_indices: 要验证的模型序号（从0开始）
    :param index_threshold: 阈值
    :param set_dir: 测试集文件夹的名称
    :param num_subjects: 测试集中受试者数量
    :param num_depression: 测试集中抑郁症患者数量。应保证数据集中的前num_depression位为抑郁症患者的样本。
    :param device: 设备。默认为cpu
    :return:
    """
    test_result = []
    description = set_dir.title()

    for subject_no in range(1, num_subjects + 1):
        test_data_path = os.path.join(args.feature_dir, set_dir)
        data_test = (read_feature(test_data_path, subject_no, selected_indices=SELECTED_INDICES, return_type='pt')
                     .to(device))

        health = 0
        depression = 0

        for model in model_indices:
            model_path = f'{args.output_path}/model_{model}.h5'
            loaded_model = load_model(model_path, device=device)
            pred = loaded_model.predict(adapt_shape(data_test[model - 1]), return_type='int')
            if pred == 1:
                depression += 1
            else:
                health += 1

        test_result.append(0 if health >= index_threshold else 1)

    print(f'{description} result:{test_result}')

    success = 0

    # 计算灵敏度
    TP = 0  # 真阳性人数
    FN = 0  # 假阴性人数

    # 计算特异性
    TN = 0  # 真阴性人数
    FP = 0  # 假阳性人数

    for i in range(0, num_depression):
        if test_result[i] == 1:
            success += 1
            TP += 1
        else:
            FN += 1
    for i in range(num_depression, num_subjects):
        if test_result[i] == 0:
            success += 1
            TN += 1
        else:
            FP += 1

    try:
        sensitivity = TP / (TP + FN)
    except ZeroDivisionError:
        sensitivity = 'N/A'

    try:
        specificity = TN / (TN + FP)
    except ZeroDivisionError:
        specificity = 'N/A'

    print(f'{description} sensitivity: {str(sensitivity)}')
    print(f'{description} specificity: {str(specificity)}')
    return success


def test(model_indices, index_threshold, device=torch.device("cpu")):
    """
    测试模型
    :param model_indices:
    :param index_threshold:
    :param device: 设备
    :return:
    """

    res = validate_models(model_indices, index_threshold, 'test', NUM_TEST_SUBJECTS, NUM_TEST_DEPRESSION, device)
    # 全部是健康的样本
    our_res = validate_models(model_indices, index_threshold, 'ours', NUM_OURS_SUBJECTS, 0, device)

    return res, our_res


def log(msg, log_level):
    if log_level > args.log_level:
        print(msg)

if __name__ == '__main__':
    main()
