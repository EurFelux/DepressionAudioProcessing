# plot feature importance manually
# _*_ coding:utf8 _*_
import numpy as np
from xgboost import XGBClassifier, plot_importance
from matplotlib import pyplot
from constants import DIM_FEATURES, NUM_FEATURES
from utils import int2str
from data_process import process_data
import os


feature_dir = '/home/wangjiyuan/data/2022data/train_enhance/test_feature'
data, _, Y_train, _ = process_data(feature_dir, split=False, to_tensor=False, selected_indices=None)
data = data.reshape(-1, NUM_FEATURES)
Y_train = Y_train.reshape(-1)
print(data.shape)
print(Y_train[:30])

# fit model no training data
model = XGBClassifier()
model.fit(data, Y_train)

# feature importance
print(model.feature_importances_)

# gain most 20 important features
indices = np.argsort(model.feature_importances_)[::-1][:20]
print(list(indices))

# plot
plot_importance(model, importance_type="gain", max_num_features=DIM_FEATURES)
pyplot.savefig('feature_gain_test_0827_test.jpg')
pyplot.show()
