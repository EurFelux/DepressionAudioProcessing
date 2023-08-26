"""
用于转换health_data_enhance文件夹下的文件名
"""

import os
from utils import int2str

dataset_dir = '/home/wangjiyuan/data/2022data/health_data_enhance'

for i in range(17, 39):
    os.remove(os.path.join(dataset_dir, str(i)))

# dirs = os.listdir(dataset_dir)
# dirs.remove('1s_features')
# [dirs.remove(dir) for dir in dirs if dir.startswith('audio_feature')]
# dirs.sort()
# num = 0
# for subject in dirs:
#     num += 1
#     subject_path = os.path.join(dataset_dir, subject)
#     print(f'Name changed from {subject} to {int2str(num)}')
#     os.rename(subject_path, os.path.join(dataset_dir, int2str(num)))
