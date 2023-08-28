# -*- coding: utf8 -*-
import os

import torch
import torch.nn as nn
import numpy as np
from ffc import FFC_BN_ACT


# 参照ECA-Net论文改为nn.Module的实现
class ECA(nn.Module):
    def __init__(self, num_channel, b=1, gamma=2):
        super(ECA, self).__init__()
        t = int(abs((np.log2(num_channel) + b) / gamma))
        kernel_size = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x):
        # x.shape: (batch_size, channel, height, width) = (N, C, H, W)
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv1(x)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.mod_seq_1 = nn.Sequential(
            SpatialAttention(),
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 1), stride=(1, 1), padding=1),
            # FFC_BN_ACT(in_channels=1, out_channels=32, kernel_size=(3, 1), stride=(1, 1), padding=1,
            #            activation_layer=nn.ReLU(), ratio_gin=0.5, ratio_gout=0.5),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 1), stride=(1, 1), padding=1),
            nn.Tanh(),
            nn.MaxPool2d((1, 2))
        )
        self.eca = ECA(32)
        self.mod_seq_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 1), stride=(1, 1), padding=1),
            nn.Tanh(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), stride=(1, 1), padding=1),
            nn.Tanh(),
            nn.MaxPool2d((1, 2)),
            nn.Flatten()
        )
        self.dense = nn.Linear(3840, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.mod_seq_1(x)
        x = x + self.eca(x)
        x = self.mod_seq_2(x)
        y = self.dense(x)
        return y

    def predict(self, x, return_type='pt'):
        """
        预测，直接给出预测结果。
        :param x: 输入，形状为(num_subjects, 1, dim_features, 1)
        :param return_type: 返回值类型，'pt'表示pytorch张量，'np'表示numpy数组, 'int'表示整数或整数列表, 'str'表示字符串。
        :return:
        """
        y_prob = self.softmax(self.forward(x))
        res = y_prob.argmax(dim=1)
        if return_type == 'pt':
            return res
        elif return_type == 'np':
            return res.detach().numpy()
        elif return_type == 'int':
            res = [1 if i == 1 else 0 for i in res]
            return res[0] if len(res) == 1 else res
        elif return_type == 'str':
            return ['depression' if i == 1 else 'health' for i in res]

    def save(self, path):
        torch.save(self.state_dict(), path)


def load_model(path, device=torch.device("cpu")):
    """
    加载模型
    :param path: 模型路径
    :param device: 设备
    :return: 模型实例
    """
    model = CNN().to(device)
    model.load_state_dict(torch.load(path))
    return model


def adapt_shape(features: torch.Tensor):
    """
    改变特征张量的形状
    :param features: 特征张量，形状为(num_subjects, dim_features)或者(dim_features)
    :return: 形状为(num_subjects, 1, dim_features, 1)的张量
    """
    if len(features.shape) == 1:
        features = features.unsqueeze(0)
    return features.unsqueeze(1).unsqueeze(3)  # 变为(num_subjects, 1, dim_features, 1)
