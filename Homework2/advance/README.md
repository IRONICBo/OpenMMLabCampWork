# Coriander Detection

This project use OpenMMLab `MMdetection` to detect coriander.The dataset is personal, using 880 coriander pictures.

![Result](https://user-images.githubusercontent.com/47499836/218019821-e6dff5e2-ae53-4072-a1b0-da7980d36ff3.png)

### 1. Label the dataset

Using [`labelme`](https://github.com/wkentaro/labelme.git) to label the dataset. The labelme is a graphical image annotation tool. It is written in Python and uses Qt for its graphical interface. It is free software, licensed under the New BSD License.

### 2. Convert the dataset to VOC2007 format

From the `labelme` label, we can get the `json` file. Using `labelme2voc.py` to convert the `json` file to `VOC2007` format.

> Because of the label doesn't contains `difficult`. It went wrong in MMdetection. So, I delete `difficult` label.

Then, we can split the dataset to train, test and val dataset.Using `gen_image_sets.py`.

### 3. Config the model

Using `yolov3_mobilenetv2_320_300e_coco.py` as the base config. Change the `num_classes` to 1. Change the `dataset_type` to VOCDataset.

### 4. Train the model

Using `https://cloud.blsc.cn/` to train the model. The model is trained for 30 epochs. And the final AP is 0.9.

### 5. Test the model

Using `webcam_detect.py` to test the model. It provide the image resource from the webcam or screen.

### Reference

1. Train log: `slurm-286514.out`
2. Model: https://www.dropbox.com/s/tzbogtltd8jtln1/coriander_yolov3_mobilenetv2.pth?dl=0