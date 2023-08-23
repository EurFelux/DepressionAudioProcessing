from data_process import read_file

train_path = r'/home/wangxu/project/Audioprocessing/2022data/train_enhance'
arr = read_file(train_path, 1)
print(arr.shape)
