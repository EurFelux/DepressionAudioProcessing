"""
一次性脚本，取出整个train_enhance，以及health_data_enhance中前3个样本，作为训练集；
取出health_data_enhance的余下样本作为测试集。
"""

import os
import shutil
from utils import int2str
from constants import NUM_TRAIN_SUBJECTS, NUM_TEST_SUBJECTS, NUM_OURS_SUBJECTS
from tqdm import tqdm

dataset_dir = '/home/wangjiyuan/data/2022data'
train_dir = os.path.join(dataset_dir, 'train_enhance')
test_dir = os.path.join(dataset_dir, 'test_enhance')
health_dir = os.path.join(dataset_dir, 'health_data_enhance')
new_train_dir = os.path.join(dataset_dir, 'new_train')
new_test_dir = os.path.join(dataset_dir, 'new_ours_test')

add_num = 3

# # 创建新的训练集
# if not os.path.exists(new_train_dir):
#     os.mkdir(new_train_dir)
# for i in tqdm(range(1, NUM_TRAIN_SUBJECTS + 1), desc="Processing train subjects", unit="subject", leave=False):
#     shutil.copytree(os.path.join(train_dir, int2str(i)), os.path.join(new_train_dir, int2str(i)), dirs_exist_ok=True)
# for i in tqdm(range(1, add_num + 1), desc="Processing health subjects", unit="subject", leave=False):
#     shutil.copytree(os.path.join(health_dir, int2str(i)), os.path.join(new_train_dir, int2str(i + NUM_TRAIN_SUBJECTS)),
#                     dirs_exist_ok=True)

# 创建新的测试集
if not os.path.exists(new_test_dir):
    os.mkdir(new_test_dir)
for i in range(add_num + 1, NUM_OURS_SUBJECTS + 1):
    shutil.copytree(os.path.join(health_dir, int2str(i)), os.path.join(new_test_dir, int2str(i)))
