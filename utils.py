# -*- coding: utf8 -*-

import random
import os
import numpy as np
import torch


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
