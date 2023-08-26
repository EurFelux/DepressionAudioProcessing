# -*- coding: UTF-8 -*-
# 包的引入 这些包python自带
import os
from subprocess import call
from constants import NUM_QUESTIONS, NUM_TRAIN_SUBJECTS, NUM_TEST_SUBJECTS
import argparse

# 路径设置
# SMILExtract_Debug.exe所在的文件路径
path_execute_file = r'/home/wangxu/project/opensmile/build/progsrc/smilextract/SMILExtract'  # 不是要求路径具体大小 省略号就是简单省略
# opensmile配置文件所在的路径  一般根据要求会选择不同的配置文件
path_config = r'/home/wangxu/project/opensmile/config/emobase/emobase2010.conf'

path_audio = r'/home/wangxu/project/Audioprocessing/2022data/test_enhance'  # E:\pythonProject\Audioprocessing\test该目录下是各个类别文件夹，类别文件夹下才是wav语音文件,比如说，我把wav文件放在了voice的文件夹里，但是voice在new文件夹里  所以应该具体到new文件夹即可，因为下面的代码是对整个文件夹里的所有文件目录里的文件进行操作，具体适用于多种不同类型的语音来进行提取特征
dir_output = r'/home/wangxu/project/Audioprocessing/2022data/test_enhance/1s_features'  # E:\pythonProject\Audioprocessing这里的路径可以自行设置比如"...\\...\\"python要加一个\转义字符


# 利用cmd调用exe文件
def opensmile(_path_execute_file, _path_config, _path_audio, _dir_output, _name):
    """
    调用命令行执行opensmile
    :param _path_execute_file: openSMILE路径
    :param _path_config: 配置
    :param _path_audio: 音频路径
    :param _dir_output: 特征输出目录
    :param _name: 特征名
    :return:
    """
    cmd = f'{_path_execute_file} -C {_path_config} -I {_path_audio} -O {_dir_output} -N {_name}'
    call(cmd, shell=True)


def extract_features(path_dataset, dir_arff, num_subjects):
    """
    提取数据集特征。
    :param path_dataset:数据集路径。路径下有若干个名为01, 02, ..., <num_subjects>的文件夹，每个文件夹下有若干个wav文件。
    此函数使用DeepFilterNet产生的文件，命名为01_DeepFilterNet.wav, 02_DeepFilterNet.wav, ..., <num_questions>_DeepFilterNet.wav。
    :param dir_arff: 特征文件保存目录
    :param num_subjects: 数据集中受试者数量
    :return:
    """
    subjects = [str(i).rjust(2, '0') for i in range(1, num_subjects + 1)]
    for subject in subjects:
        subject_path = os.path.join(path_dataset, subject)
        assert os.path.isdir(subject_path)
        for i in range(1, NUM_QUESTIONS + 1):
            wav_name = f'{str(i).rjust(2, "0")}_DeepFilterNet.wav'
            assert os.path.exists(os.path.join(subject_path, wav_name))
            file_path = os.path.join(subject_path, wav_name)
            output_name = f'audio_feature{str(i).rjust(2, "0")}.txt'
            output_path = os.path.join(dir_arff, output_name)
            opensmile(path_execute_file, path_config, file_path, output_path,
                      f'{subject}-{str(i).rjust(2, "0")}_DeepFilterNet')


if __name__ == '__main__':
    extract_features(path_audio, dir_output, NUM_TRAIN_SUBJECTS)
