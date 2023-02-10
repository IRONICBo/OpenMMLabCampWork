# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import cv2
import torch

from mmdet.apis import inference_detector, init_detector
# 用于捕获桌面图像
from PIL import ImageGrab # Windows/Ubuntu
import numpy as np


# ython webcam_demo.py --device cpu demo/coriander_yolov3_mobilenetv2.py demo/latest.pth --use-screen 1
def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    parser.add_argument(
        '--use-screen', type=int, default=0, help='use screen capture(default not to use screen capture)')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    device = torch.device(args.device)

    model = init_detector(args.config, args.checkpoint, device=device)

    # 获取桌面的图像
    if args.use_screen == 0:
        camera = cv2.VideoCapture(args.camera_id)
    print('Press "Esc", "q" or "Q" to exit.')
    while True:
        start_time = cv2.getTickCount()
        
        if args.use_screen != 0:
            image = ImageGrab.grab()
            img = np.array(ImageGrab.grab(bbox=(500,200,1200,900))) # bbox x, y, width, height
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            ret_val, img = camera.read()
        
        result = inference_detector(model, img)

        end_time = cv2.getTickCount()
        print("FPS: ", cv2.getTickFrequency() / (end_time - start_time))

        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break

        model.show_result(
            img, result, score_thr=args.score_thr,thickness=5, font_size=18, wait_time=0.01, show=True)


if __name__ == '__main__':
    main()
