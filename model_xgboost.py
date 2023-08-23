# plot feature importance manually
# _*_ coding:utf8 _*_
import numpy as np
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot

import os

def readFile_byindex(index,path,i):
    txt_path = path + 'audio_feature0{}.txt'.format(i)
    #mat_path = path + 'whisper_features0{}.mat'.format(i)
    f = open(txt_path)
    lines = f.readlines()
    del lines[0:1589]
    first_ele = True
    for data in lines:
        data = data.strip('\n')
        nums = data.split(',')
        if first_ele:
            matrix = np.array(nums)
            first_ele = False
        else:
            matrix = np.c_[matrix,nums]
    matrix = matrix.transpose()
    #b = [1403,227,1493,290,160,154,1249,97,572,45,89,187,621,334,207,844,608,396,295,205]
    result = [float(matrix[index-1][c]) for c in range(1,1583)]
    
    #mat = hdf5.loadmat(mat_path)['dataset_features'][index-1]
    #features_mat = [float(mat[i]) for i in range(1280)]
    #result = result + features_mat
    
    f.close()
    
    return result

data = []
Y_train = []
for m in range(1,30):
    #data = []
    #Y_train = []
    for i in range(1,39):
        train_data = readFile_byindex(m,'/home/wangxu/project/Audioprocessing/2022data/train_enhance/',i)
        data.append(train_data)
        if i < 17:
            Y_train.append(1)
        elif i>=17:
            Y_train.append(0)
    #for i in range(29,58):
    #    train_data = readFile_byindex(m,'/data/audio_data/health_data_enhance/',i)
    #    data.append(train_data)
    #    Y_train.append(0)
    #for i in range(36,72):
    #    train_data = readFile_byindex(m,'/data/audio_data/hospital_data_enhance/',i)
    #    data.append(train_data)
    #    Y_train.append(1)

data_1 = np.array(data)
print(data_1.shape)

#
#
# # fit model no training data
model = XGBClassifier()
model.fit(data_1, Y_train)
# feature importance
print(model.feature_importances_)
# plot
plot_importance(model,importance_type="gain",max_num_features=20)
pyplot.savefig('feature_gain_test_0729.jpg')
pyplot.show()