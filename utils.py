# -*- coding: utf8 -*-

import random
import os
import numpy as np
import torch
from constants import MAX_DIGITS


def set_seeds(seed):
    """
    设置随机种子
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return None


def int2str(num):
    """
    将整数转换为字符串，不足MAX_DIGITS位的在前面补0，超过的截断
    :param num: 整数
    :return: 转换后的字符串
    """
    return str(num).rjust(MAX_DIGITS, '0')