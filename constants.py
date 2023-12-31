TO_DEL_LINES = 1589
"""
opensmile提取的文件，前1589行是特征说明，1590行开始是特征信息。
"""

NUM_QUESTIONS = 29
"""
问题个数。
"""

NUM_FEATURES = 1582
"""
问题的特征维度。
"""

DIM_FEATURES = 20
"""
选择后特征向量的特征维度
"""

NUM_TRAIN_SUBJECTS = 41
"""
训练集的受试者数量。
"""

NUM_TRAIN_DEPRESSION = 16
"""
训练集中的抑郁症受试者数量。前NUM_TRAIN_DEPRESSION个样本为抑郁症患者，后面为健康受试者。
"""

NUM_TEST_SUBJECTS = 13
"""
测试集的受试者数量。
"""

NUM_TEST_DEPRESSION = 6
"""
测试集中的抑郁症受试者数量。前NUM_TEST_DEPRESSION个样本为抑郁症患者，后面为健康受试者。
"""

NUM_OURS_SUBJECTS = 13
"""
我们的数据集中的受试者数量。
"""

NUM_OURS_DEPRESSION = 0
"""
我们的数据集中的抑郁症受试者数量。前NUM_OURS_DEPRESSION个样本为抑郁症患者，后面为健康受试者。
"""

MAX_DIGITS = 2
"""
数据集中受试者序号的最大位数。
"""

# SELECTED_INDICES = [184, 226, 1402, 1490, 317, 289, 206, 416, 107, 44, 467, 1062, 684, 620, 1035, 51, 175, 1172, 112,
                    # 118]
SELECTED_INDICES = [751, 289, 226, 1490, 44, 616, 321, 111, 354, 1173, 521, 14, 52, 1069, 764, 163, 569, 96, 1548, 874]

"""
选择的特征序号
"""

SPLIT_RATE = 0.25
"""
分割数据集时的参数，该值表示验证集所占的比例。
"""

OPENSMILE_LOG_LEVEL = 0
"""
调用opensmile提取特征时的日志等级，设置为0表示静默。默认为2
"""

LOG_LEVEL = 2
"""
调用main.py时的默认日志等级。
"""