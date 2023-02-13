import os
import random

IMAGES_PATH = 'tutorial/images'
SEGMAPS_PATH = 'tutorial/masks'

# 打乱数据表
all_file_list = os.listdir(IMAGES_PATH)
all_file_num = len(all_file_list)
random.shuffle(all_file_list)

train_ratio = 0.8
test_ratio = 1 - train_ratio

# 划分数据集
train_file_list = all_file_list[:int(all_file_num*train_ratio)]
test_file_list = all_file_list[int(all_file_num*train_ratio):]

os.mkdir('tutorial/splits')
with open('tutorial/splits/train.txt', 'w') as f:
    f.writelines(line.split('.')[0] + '\n' for line in train_file_list)
with open('tutorial/splits/val.txt', 'w') as f:
    f.writelines(line.split('.')[0] + '\n' for line in test_file_list)