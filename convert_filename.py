"""
一次性，用于改变文件命名
"""

import os
from utils import int2str

dataset_dir = '/home/wangjiyuan/data/2022data/new_ours_test'

# 04~16改为01~13
for i in range(4, 17):
    path = os.path.join(dataset_dir, f'{int2str(i)}')
    print(f'Name changed from {int2str(i)} to {int2str(i - 3)}')
    os.rename(path, os.path.join(dataset_dir, f'{int2str(i - 3)}'))

# dirs = os.listdir(dataset_dir)
# dirs.remove('1s_features')
# [dirs.remove(dir) for dir in dirs if dir.startswith('audio_feature')]
# dirs.sort()
# num = 0
# for subject in dirs:
#     num += 1
#       subject_path = os.path.join(dataset_dir, subject)
#     print(f'Name changed from {subject} to {int2str(num)}')
#     os.rename(subject_path, os.path.join(dataset_dir, int2str(num)))
