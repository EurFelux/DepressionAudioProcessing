from data_process import process_data

train_path = r'/home/wangjiyuan/data/2022data/train_enhance'

train_features, val_features, train_labels, val_labels = process_data(train_path, split=True, to_tensor=False)

print(train_features.shape)
print(val_features.shape)
print(train_labels.shape)
print(val_labels.shape)
