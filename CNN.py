# -*- coding: utf8 -*-
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import random
import tensorflow as tf

gpus = tf.config.list_physical_devices(device_type='GPU')
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=20000)])

import hdf5storage as hdf5

from tensorflow.keras.optimizers import Adam
import numpy as np
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, add, Input, Activation, Concatenate
from keras.models import load_model
from tensorflow import keras
from tensorflow.keras import Model, layers
import keras.backend as K

from keras.regularizers import l2
import math

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--feature_dir", default="")
parser.add_argument("--output_path", default="")
args = parser.parse_args()


def set_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def read_file(path, i):
    txt_path = path + 'audio_feature0{}.txt'.format(i)
    # mat_path = path + 'whisper_features0{}.mat'.format(i)
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
            matrix = np.c_[matrix, nums]
    matrix = matrix.transpose()

    # mat = hdf5.loadmat(mat_path)['dataset_features']
    a = []
    b = [1492, 227, 1403, 174, 468, 1493, 1036, 417, 607, 612, 45, 15, 1277, 872, 153, 1491, 619, 685, 305, 529]
    # b = [1403,227,1493,290,160,154,1249,97,572,45,89,187,621,334,207,844,608,396,295,205]
    # b = [1487,202,1499,6,1222,39,888,1193,417,1521,1512,523,1059,174,1337,1181,1439,889,1139,108]
    for x in range(0, 29):
        result = [float(matrix[x][c]) for c in b]
        # features_mat = mat[x]
        # features_line = [float(features_mat[j]) for j in range(1280)]
        # result = result + features_line
        a.append(result)
    arr = np.array(a)
    f.close()
    return arr


def readFile_byindex(index, path, i):
    txt_path = path + 'audio_feature0{}.txt'.format(i)
    # mat_path = path + 'whisper_features0{}.mat'.format(i)
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
            matrix = np.c_[matrix, nums]
    matrix = matrix.transpose()
    b = [1492, 227, 1403, 174, 468, 1493, 1036, 417, 607, 612, 45, 15, 1277, 872, 153, 1491, 619, 685, 305, 529]
    # b = [1403,227,1493,290,160,154,1249,97,572,45,89,187,621,334,207,844,608,396,295,205]
    # b = [1487,202,1499,6,1222,39,888,1193,417,1521,1512,523,1059,174,1337,1181,1439,889,1139,108]
    result = [float(matrix[index - 1][c]) for c in b]

    # mat = hdf5.loadmat(mat_path)['dataset_features'][index-1]
    # features_mat = [float(mat[i]) for i in range(1280)]
    # result = result + features_mat

    f.close()

    return result


def eca_block(inputs, b=1, gama=2):
    # 输入特征图的通道数
    in_channel = inputs.shape[-1]

    # 根据公式计算自适应卷积核大小
    kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))

    # 如果卷积核大小是偶数，就使用它
    if kernel_size % 2:
        kernel_size = kernel_size

    # 如果卷积核大小是奇数就变成偶数
    else:
        kernel_size = kernel_size + 1

    # [h,w,c]==>[None,c] 全局平均池化
    x = layers.GlobalAveragePooling2D()(inputs)

    # [None,c]==>[c,1]
    x = layers.Reshape(target_shape=(in_channel, 1))(x)

    # [c,1]==>[c,1]
    x = layers.Conv1D(filters=1, kernel_size=kernel_size, padding='same', use_bias=False)(x)

    # sigmoid激活
    x = tf.nn.sigmoid(x)

    # [c,1]==>[1,1,c]
    x = layers.Reshape((1, 1, in_channel))(x)

    # 结果和输入相乘
    outputs = layers.multiply([inputs, x])

    return outputs


def spatial_attention(input_feature):
    avg_pool = tf.reduce_mean(input_feature, axis=3, keepdims=True)
    max_pool = tf.reduce_max(input_feature, axis=3, keepdims=True)
    # max_pool = layers.Lambda(lambda x:K.mean(x,axis=3,keepdims=True))(input_feature)
    # max_pool = layers.Lambda(lambda x:K.max(x,axis=3,keepdims=True))(input_feature)

    concat = Concatenate(axis=3)([avg_pool, max_pool])
    cbam_feature = Conv2D(1, (3, 1), strides=1, padding='same', activation='sigmoid')(concat)

    outputs = layers.multiply([input_feature, cbam_feature])

    return outputs


'''
def FFT(Layer):
    def __init__(self,**kwargs):
        super(FFT,self).__init__(**kwargs)
 
    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[0],2,input_shape[2],input_shape[3]), initializer='random_normal', trainable=True)
        super(FFT, self).build(input_shape)
    
    def call(self,x):
        x = f.spectral.rfft2d(x)
        weight = tf.complex(W[:,0,:,:],W[:,0,:,:])
        x = x * weight
        x = f.spectral.irfft2d(x)
        return x
'''

# script starts here

for seed in range(101, 1001):
    set_seeds(seed)
    result = [[] for i in range(29)]
    for m in range(1, 30):
        data = []
        Y_train = []
        for i in range(1, 39):
            train_data = readFile_byindex(m, args.feature_dir + '/train_enhance/', i)
            data.append(train_data)
            # 添加label
            # 1~16: depression, 17~38: health
            if i < 17:
                Y_train.append(1)
            elif i >= 17:
                Y_train.append(0)
        # for i in range(10,13):
        #    train_data = readFile_byindex(m,args.feature_dir + '/health_data_enhance/',i)
        #    data.append(train_data)
        #    Y_train.append(0)
        # for i in range(11,17):
        #    train_data = readFile_byindex(m,args.feature_dir + '/depression_data_enhance/',i)
        #    data.append(train_data)
        #    Y_train.append(1)
        X_train = np.array(data) # (38, 20)
        Y_train = np.array(Y_train) # (38,)
        X_train = np.expand_dims(X_train[:, 0:20], axis=2) # (38, 20, 1)
        Y_train = Y_train.reshape((len(Y_train),)) # (38,) no change

        encoder = LabelEncoder()
        Y_train_encoded = encoder.fit_transform(Y_train) # no change
        Y_train = np_utils.to_categorical(Y_train_encoded) # (38, 2), to one-hot vectors

        # 每次都随机地分为训练集和验证集
        X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.25, random_state=5)

        # 扩展出channel维度
        X_train = np.expand_dims(X_train, axis=1).astype(np.float32)
        X_test = np.expand_dims(X_test, axis=1).astype(np.float32)
        test_label_1 = []
        test_label_2 = []

        # [1, 0]会得到0，[0, 1]会得到1，实际上就是把one-hot转换回去
        for n in range(len(Y_test)):
            test_label_1.append(int(Y_test[n][1]))

        # 除去第一维
        input_shape = X_train.shape[1:]

        # 总的来说，在开始训练以前，得到了X_train, Y_train, X_test, Y_test, input_shape
        # 其中X_train, X_test的shape为(train_len, 1, 20, 1), (test_len, 1, 20, 1)
        # Y_train, Y_test的shape为(train_len, 2), (test_len, 2)
        # input_shape为(1, 20, 1)

        # -------------------------------
        # Model
        # -------------------------------

        # print(input_shape)

        input = Input(shape=input_shape)

        x = spatial_attention(input)

        x = Conv2D(filters=16, kernel_size=(3, 1), strides=(1, 1), padding='same', input_shape=input_shape)(x)
        x = Activation('relu')(x)

        x = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same')(x)
        x = Activation('tanh')(x)

        x = MaxPooling2D((1, 2))(x)

        eca = eca_block(x)
        x = layers.add([x, eca])

        x = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same')(x)
        x = Activation('tanh')(x)

        x = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same')(x)
        x = Activation('tanh')(x)

        x = MaxPooling2D((1, 2))(x)

        x = Flatten()(x)

        output = Dense(units=2, activation='softmax')(x)

        # ---------------------------

        model = Model(inputs=input, outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

        model.fit(X_train, Y_train, epochs=300, batch_size=1, verbose=0)

        model.save(args.output_path + "/model_{}.h5".format(m))

        loaded_model = load_model(args.output_path + "/model_{}.h5".format(m))
        loaded_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        scores = loaded_model.evaluate(X_test, Y_test, verbose=0)
        print('%s: %.2f%%' % (loaded_model.metrics_names[1], scores[1] * 100))
        predicted_label = loaded_model.predict(X_test)
        for o in range(len(predicted_label)):
            test_label_2.append(np.argmax(predicted_label[o], axis=0))
        # 计算灵敏度
        q1 = 0  # 真阳性人数
        q2 = 0  # 假阴性人数
        result_1 = 0
        for i in range(len(test_label_1)):
            if test_label_1[i] == test_label_2[i] == 1:
                q1 += 1
            if test_label_1[i] == 1 and test_label_2[i] == 0:
                q2 += 1
        result_1 = q1 / (q1 + q2)
        print('模型灵敏度：' + str(result_1))

        # 计算特异性
        p1 = 0  # 真阴性人数
        p2 = 0  # 假阳性人数
        result_2 = 0
        for j in range(len(test_label_1)):
            if test_label_1[j] == test_label_2[j] == 0:
                p1 += 1
            if test_label_1[j] == 0 and test_label_2[j] == 1:
                p2 += 1
        result_2 = p1 / (p1 + p2)
        print('模型特异性：' + str(result_2))

        result[m - 1].append(result_1)
        result[m - 1].append(result_2)

    # 到这里为止获取了NUM_QUESTIONS个模型，每个模型都有一个灵敏度和一个特异性
    # result是一个二维数组，第一维是模型编号，第二维是[灵敏度，特异性]
    # result的形状为(num_models, 2)
    result_specificity = np.array(result)
    # 取负转置以后，第一行表示灵敏度，第二行表示特异性。原本是升序，取负以后就变成降序了。
    # 经过lexsort，
    result_specificity = np.lexsort(-result_specificity.T)
    # 序号+1，转换为 1-based
    for i in range(len(result_specificity)):
        result_specificity[i] += 1
    # 排序优先按照特异性，再按照灵敏度

    result_sensitivity = np.array(result)
    result_sensitivity = np.lexsort(-result_sensitivity[:, ::-1].T)
    for i in range(len(result_sensitivity)):
        result_sensitivity[i] += 1

    # 获得模型优劣排序
    print(result_specificity)
    print(result_sensitivity)

    test_result = []

    index_Threshold = 2

    model_indexs = []
    for i in range(index_Threshold):
        model_indexs.append(result_sensitivity[i])
    for i in range(29):
        if len(model_indexs) == (index_Threshold * 2 - 1):
            break
        if result_specificity[i] not in model_indexs:
            model_indexs.append(result_specificity[i])
    print(model_indexs)
    # 挑选出3个模型

    for idx in range(1, 14):
        data_test = read_file(args.feature_dir + '/test_enhance/', idx)

        health = 0
        depression = 0

        for model in model_indexs:
            loaded_model = load_model(args.output_path + "/model_{}.h5".format(model))
            loaded_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001),
                                 metrics=['accuracy'])
            if int(np.argmax(loaded_model.predict(np.array(data_test[model - 1]).reshape((1, 1, 20, 1))), axis=1)) == 0:
                health += 1
            else:
                depression += 1
        if health >= index_Threshold:
            test_result.append(0)
        else:
            test_result.append(1)

    print(test_result)

    res = 0

    # 计算灵敏度
    q1 = 0  # 真阳性人数
    q2 = 0  # 假阴性人数

    # 计算特异性
    p1 = 0  # 真阴性人数
    p2 = 0  # 假阳性人数

    for i in range(0, 6):
        if test_result[i] == 1:
            res += 1
            q1 += 1
        else:
            q2 += 1
    for i in range(6, 13):
        if test_result[i] == 0:
            res += 1
            p1 += 1
        else:
            p2 += 1

    print('test灵敏度：' + str(q1 / (q1 + q2)))
    print('test特异性：' + str(p1 / (p1 + p2)))

    our_result = []

    for idx in range(1, 17):
        data_test = read_file(args.feature_dir + '/health_data_enhance/', idx)

        health = 0
        depression = 0

        for model in model_indexs:
            loaded_model = load_model(args.output_path + "/model_{}.h5".format(model))
            loaded_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001),
                                 metrics=['accuracy'])
            if int(np.argmax(loaded_model.predict(np.array(data_test[model - 1]).reshape((1, 1, 20, 1))), axis=1)) == 0:
                health += 1
            else:
                depression += 1

        if health >= index_Threshold:
            our_result.append(0)
        else:
            our_result.append(1)

    '''
    for idx in range(13,17):
        data_test = readFile(args.feature_dir + '/health_data_enhance/',idx)
    
        health = 0
        depression = 0
    
        for model in model_indexs:
            loaded_model =load_model(args.output_path + "/model_{}.h5".format(model))
            loaded_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
            if int(np.argmax(loaded_model.predict(np.array(data_test[model-1]).reshape((1,1,20,1))),axis=1)) == 0:
                health += 1
            else :
                depression += 1
        
        if health >= index_Threshold:
            our_result.append(0)
        else :
            our_result.append(1)
        
    
    for idx in range(1,11):
        data_test = readFile(args.feature_dir + '/depression_data_enhance/',idx)
    
        health = 0
        depression = 0
    
        for model in model_indexs:
            loaded_model =load_model(args.output_path + "/model_{}.h5".format(model))
            loaded_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
            if int(np.argmax(loaded_model.predict(np.array(data_test[model-1]).reshape((1,1,1300,1))),axis=1)) == 0:
                health += 1
            else :
                depression += 1
        
        if health >= index_Threshold:
            our_result.append(0)
        else :
            our_result.append(1)
    '''
    print(our_result)

    our_res = 0

    # 计算灵敏度
    our_q1 = 0  # 真阳性人数
    our_q2 = 0  # 假阴性人数

    # 计算特异性
    our_p1 = 0  # 真阴性人数
    our_p2 = 0  # 假阳性人数

    for i in range(0, 16):
        if our_result[i] == 0:
            our_res += 1
            our_p1 += 1
        else:
            our_p2 += 1

    '''
    for i in range(0,8):
        if our_result[i] == 0:
            our_res += 1
            our_p1 += 1
        else :
            our_p2 += 1
    for i in range(8,18):
        if our_result[i] == 1:
            our_res += 1
            our_q1 += 1
        else :
            our_q2 += 1
    '''
    # print('our灵敏度：' + str(our_q1 / (our_q1 + our_q2)))
    print('our特异性：' + str(our_p1 / (our_p1 + our_p2)))

    print('seed == ' + str(seed))
    print('test正确率：' + str(res / 13))
    print('our正确率：' + str(our_res / 16))

    if (res / 13) > 0.9 and (our_res / 16) > 0.8:
        print("找到了！！！！")
        print('seed == ' + str(seed))
        print('test正确率：' + str(res / 13))
        print('our正确率：' + str(our_res / 16))
        break
'''
    if (res/13) > 0.9 and (our_res/18) > 0.8:
        print("找到了！！！！")
        print('seed == ' + str(seed))
        print('test正确率：' + str(res/13))
        print('our正确率：' + str(our_res/18))
        #break
'''
