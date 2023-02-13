# OpenMMLabCampWork
This repo is for OpenMMLabCamp homework

### Work1

##### base
- resnet152_b16_flower.py => config
- slurm-283611.out => train and test log
- model => https://www.dropbox.com/s/c0baadj3oi1mz0z/resnet152_b16_flower.pth?dl=0

Final: Epoch(val) [100][18]	accuracy_top-1: 97.3776

##### advance

Test mmcls in cifar10 dataset.

- resnet18_8xb16_cifar10.py => config
- slurm-289406.out => train and test log
- model => https://www.dropbox.com/s/j0vdegintlwec09/resnet18_8xb16_cifar10.pth?dl=0

Final: Epoch(val) [50][79]	accuracy_top-1: 94.9300, accuracy_top-5: 99.8200


### Work2

##### base

- origin2coco.py => convert dataset json file to coco json file
- mask_rcnn_r50_fpn_mstrain-poly_3x_coco.py => model config file
- color_splash.py => generate ballon splash video
- slurm-289750.out => train and test log
- out.mp4 => result video
- model => https://www.dropbox.com/s/e8yndkg94jnl04r/mask_rcnn_r50_fpn_mstrain-poly_3x_coco.pth?dl=0

Final: Epoch(val) [20][13]	bbox_mAP: 0.7868, bbox_mAP_50: 0.8989, bbox_mAP_75: 0.8787, bbox_mAP_s: 0.3535, bbox_mAP_m: 0.7177, bbox_mAP_l: 0.8422, bbox_mAP_copypaste: 0.7868 0.8989 0.8787 0.3535 0.7177 0.8422, segm_mAP: 0.8187, segm_mAP_50: 0.8813, segm_mAP_75: 0.8813, segm_mAP_s: 0.1010, segm_mAP_m: 0.7314, segm_mAP_l: 0.8702, segm_mAP_copypaste: 0.8187 0.8813 0.8813 0.1010 0.7314 0.8702

##### advance

- coriander_yolov3_mobilenetv2.py => model config file
- gen_image_sets.py => split dataset to train and test
- labelme2voc.py => convert labelme json file to voc xml file
- webcam_detect.py => detect coriander in webcam or screen
- slurm-286514.out => train and test log
- model => https://www.dropbox.com/s/tzbogtltd8jtln1/coriander_yolov3_mobilenetv2.pth?dl=0

Final: Epoch(val) [30][176]	mAP: 0.9091, AP50: 0.9090

### Work3

##### base

- pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py => model config file
- split_dataset.py => split dataset to train and test
- modify_model.py => modify model to fit my dataset
- pspnet_runnable.py => train model
- model => https://www.dropbox.com/s/ckogg0fw04wc609/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.pth?dl=0

Final: Iter(test) [515/515]  aAcc: 99.1100  mIoU: 77.5000  mAcc: 79.5200

##### advance

- pspnet_r50-d8_512x512_20k_voc.py => model config file
- split_dataset.py => split dataset to train and test
- model => https://www.dropbox.com/s/ckogg0fw04wc609/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.pth?dl=0

Final: Iter(test) [515/515]  aAcc: 99.1100  mIoU: 77.5000  mAcc: 79.5200