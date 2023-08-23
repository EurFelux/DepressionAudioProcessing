# -*- coding: utf8 -*-
import torch.nn as nn
import numpy as np

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

# TODO: 实现CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            SpatialAttention(),
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 1), stride=(1, 1), padding=1),
        )

    def forward(self, x):
        x = self.net(x)
        return x



