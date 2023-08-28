"""
一次性，用于改变文件命名
"""

import os
from utils import int2str

dataset_dir = '/home/wangjiyuan/data/2022data/test_enhance/12'

# 16~30改成15~29
for i in range(16, 31):
    path = os.path.join(dataset_dir, f'{int2str(i)}_DeepFilterNet.wav')
    print(f'Name changed from {int2str(i)}_DeepFilterNet.wav to {int2str(i - 1)}_DeepFilterNet.wav')
    os.rename(path, os.path.join(dataset_dir, f'{int2str(i - 1)}_DeepFilterNet.wav'))

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
