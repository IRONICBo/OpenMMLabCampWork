#!/usr/bin/env python

# 生成ImageSets文件夹下的train.txt和val.txt文件
import os
import random

trainval_percent = 0.8
train_percent = 0.9
xmlfilepath = 'coriander_voc/Annotations'
txtsavepath = 'coriander_voc/ImageSets/Main'
if not os.path.exists(txtsavepath):
    os.makedirs(txtsavepath)

total_xml = os.listdir(xmlfilepath)

num=len(total_xml)
list=range(num)
tv=int(num*trainval_percent)
tr=int(tv*train_percent)

trainval= random.sample(list,tv)
train=random.sample(trainval,tr)

ftrainval = open('coriander_voc/ImageSets/Main/trainval.txt', 'w')
ftest = open('coriander_voc/ImageSets/Main/test.txt', 'w')
ftrain = open('coriander_voc/ImageSets/Main/train.txt', 'w')
fval = open('coriander_voc/ImageSets/Main/val.txt', 'w')

for i  in list:
    name=total_xml[i][:-4]+'\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest .close()