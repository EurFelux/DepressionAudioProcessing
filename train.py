# -*- coding: utf8 -*-
import os
import argparse
from utils import set_seeds

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

parser = argparse.ArgumentParser()
parser.add_argument("--feature_dir", default="")
parser.add_argument("--output_path", default="")
args = parser.parse_args()


def train(seed):
    set_seeds(seed)
    NUM_SUBJECTS = 29

    result = [[] for i in range(NUM_SUBJECTS)]
    for i in range(1, NUM_SUBJECTS + 1):
        data = []
        Y_train = []
        for i in range(1, ):