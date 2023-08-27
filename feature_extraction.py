# -*- coding: UTF-8 -*-
# 包的引入 这些包python自带
import os
import argparse
from subprocess import call
from constants import NUM_QUESTIONS, NUM_TRAIN_SUBJECTS, NUM_TEST_SUBJECTS, MAX_DIGITS, OPENSMILE_LOG_LEVEL, \
    NUM_OURS_SUBJECTS
from utils import int2str
from tqdm import tqdm

# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument("--num_subjects", default=NUM_TRAIN_SUBJECTS)
parser.add_argument("--path_audio", default="/home/wangjiyuan/data/2022data/train_enhance")
parser.add_argument("--dir_output", default="/home/wangjiyuan/dev/DepressionAudioProcessing/features/train")
args = parser.parse_args()

# 路径设置
# SMILExtract_Debug.exe所在的文件路径
path_execute_file = r'/home/wangxu/project/opensmile/build/progsrc/smilextract/SMILExtract'  # 不是要求路径具体大小 省略号就是简单省略
# opensmile配置文件所在的路径  一般根据要求会选择不同的配置文件
path_config = r'/home/wangxu/project/opensmile/config/emobase/emobase2010.conf'


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

    cmd = f'{_path_execute_file} -C {_path_config} -l {OPENSMILE_LOG_LEVEL} -I {_path_audio} -O {_dir_output} -N {_name}'
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
    if not os.path.exists(dir_arff):
        os.mkdir(dir_arff)
    subjects = [int2str(i) for i in range(1, num_subjects + 1)]
    for subject_no in tqdm(subjects, desc='Processing subject', unit='subject', leave=True):
        subject_path = os.path.join(path_dataset, subject_no)
        assert os.path.isdir(subject_path)
        output_name = f'audio_feature{int2str(subject_no)}.txt'
        output_path = os.path.join(dir_arff, output_name)
        if os.path.exists(output_path):
            os.remove(output_path)
        for question_no in tqdm(range(1, NUM_QUESTIONS + 1), desc='Processing question', unit='question', leave=False):
            wav_name = f'{int2str(question_no)}_DeepFilterNet.wav'
            assert os.path.exists(os.path.join(subject_path, wav_name))
            file_path = os.path.join(subject_path, wav_name)
            opensmile(path_execute_file, path_config, file_path, output_path,
                      f'{subject_no}-{int2str(question_no)}_DeepFilterNet')


if __name__ == '__main__':
    path_audio = args.path_audio
    dir_output = args.dir_output
    num_subjects = int(args.num_subjects)
    print(num_subjects)
    extract_features(path_audio, dir_output, num_subjects)
