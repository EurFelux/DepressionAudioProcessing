# plot feature importance manually
# _*_ coding:utf8 _*_
import numpy as np
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
from constants import TO_DEL_LINES, NUM_FEATURES, NUM_QUESTIONS, NUM_TRAIN_SUBJECTS, NUM_TRAIN_DEPRESSION

import os


def read_full_feature(question_no, path, subject_no):
    feature_filename = f'audio_feature{str(subject_no).rjust(2, "0")}.txt'
    txt_path = os.path.join(path, feature_filename)
    assert os.path.exists(txt_path)
    # mat_path = path + 'whisper_features0{}.mat'.format(i)
    f = open(txt_path)
    lines = f.readlines()
    del lines[0:TO_DEL_LINES]
    first_ele = True
    for data in lines:
        data = data.strip('\n')
        nums = data.split(',')
        if first_ele:
            matrix = np.array(nums)
            first_ele = False
        else:
            matrix = np.c_[matrix, nums]
    matrix = matrix.transpose()
    # b = [1403,227,1493,290,160,154,1249,97,572,45,89,187,621,334,207,844,608,396,295,205]
    result = [float(matrix[question_no - 1][c]) for c in range(1, NUM_FEATURES + 1)]

    # mat = hdf5.loadmat(mat_path)['dataset_features'][index-1]
    # features_mat = [float(mat[i]) for i in range(1280)]
    # result = result + features_mat

    f.close()

    return result


data = []
for question_no in range(1, NUM_QUESTIONS + 1):
    for subject_no in range(1, NUM_TRAIN_SUBJECTS + 1):
        train_data = read_full_feature(question_no, '/home/wangxu/project/Audioprocessing/2022data/train_enhance/', subject_no)
        data.append(train_data)
    Y_train = np.array([1 if i <= NUM_TRAIN_DEPRESSION else 0 for i in range(1, NUM_TRAIN_SUBJECTS + 1)])

data_1 = np.array(data)
print(data_1.shape)

# # fit model no training data
model = XGBClassifier()
model.fit(data_1, Y_train)
# feature importance
print(model.feature_importances_)
# plot
plot_importance(model, importance_type="gain", max_num_features=20)
pyplot.savefig('feature_gain_test_0729.jpg')
pyplot.show()
