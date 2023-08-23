# -*- coding: UTF-8 -*-
#包的引入 这些包python自带
import os
from subprocess import call
import soundfile
#路径设置
#SMILExtract_Debug.exe所在的文件路径
pathExcuteFile = r'/home/wangxu/project/opensmile/build/progsrc/smilextract/SMILExtract'#不是要求路径具体大小 省略号就是简单省略
#opensmile配置文件所在的路径  一般根据要求会选择不同的配置文件
pathConfig = r'/home/wangxu/project/opensmile/config/emobase/emobase2010.conf'
pathAudio = r'/home/wangxu/project/Audioprocessing/2022data/test_enhance' #E:\pythonProject\Audioprocessing\test该目录下是各个类别文件夹，类别文件夹下才是wav语音文件,比如说，我把wav文件放在了voice的文件夹里，但是voice在new文件夹里  所以应该具体到new文件夹即可，因为下面的代码是对整个文件夹里的所有文件目录里的文件进行操作，具体适用于多种不同类型的语音来进行提取特征
pathOutput = r'/home/wangxu/project/Audioprocessing/2022data/test_enhance/1s_features'#E:\pythonProject\Audioprocessing这里的路径可以自行设置比如"...\\...\\"python要加一个\转义字符
#利用cmd调用exe文件
def excuteCMD(_pathExcuteFile,_pathConfig,_pathAudio,_pathOutput,_Name):
    cmd = _pathExcuteFile + " -C " + _pathConfig + " -I " + _pathAudio + " -O " + _pathOutput + " -N " + _Name
    call(cmd, shell=True)

def loopExcute(pathwav,patharff):
    number = 0 # 子目录，对目录里所有wav目标文件进行处理
    categories = os.listdir(pathwav)
    categories.sort()
    for category in categories:
        category_path = os.path.join(pathwav,category)
        #print(category)
        if os.path.isdir(category_path) and len(category)<3: # len(category)<3 什么意思？
            #print(category_path)
            number += 1
            for i in range(1,30):
                for f in os.listdir(category_path):
                #for i in range(1,30):
                    if f == f"{str(i).rjust(2,'0')}_DeepFilterNet.wav":
                        #print(f)
                        wavs, fs = soundfile.read(os.path.join(category_path,f))
                        print(fs)
                        file_path = os.path.join(category_path,f)
                        num = int(wavs.shape[0]/(16000*1))+1 # 神秘公式
                        print(num)
                        for j in range(num):
                            for f2 in os.listdir(category_path):
                                if f2 == f'{str(i)}_1s_{str(j+1)}.wav':
                                    print(f2)
                                    file_path = os.path.join(category_path,f2)
                                    wavs, fs = soundfile.read(file_path)
                                    wavs = wavs.reshape(1,-1)
                                    if len(wavs[0])==16000:
                                        name = category + '-' + os.path.splitext(f2)[0]
                                        #outputname = 'audio_feature0{}.txt'.format(i)#这里是将所有的特征文件写道一个arff文件里，也可以用一个一直在变的名称来实现一个语音对应一个特征文件
                                        outputname = 'audio_1s_features_{}.txt'.format(str(number).rjust(2,'0'))
                                        output_path = os.path.join(patharff,outputname)
                                        excuteCMD(pathExcuteFile, pathConfig, file_path, output_path, name)

if __name__ == '__main__':
    #excuteCMD(pathExcuteFile, pathConfig, pathAudio, pathOutput)
    loopExcute(pathAudio, pathOutput)