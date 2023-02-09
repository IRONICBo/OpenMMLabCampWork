import mmcv
from mmdet.apis import init_detector, inference_detector
import os
import numpy as np
from tqdm import tqdm

CONFIG = 'mask_rcnn_r50_fpn_mstrain-poly_3x_coco.py'
CHECKPOINT = 'latest.pth'

# 获取当前路径
path = os.getcwd()

video = mmcv.VideoReader('./test_video.mp4')
mask_rcnn_model = init_detector(CONFIG, CHECKPOINT)

video.cvt2frames('out_dirs') # 转换成视频序列

# 列举出所有的图片  
img_list = mmcv.utils.scandir('out_dirs', '.jpg', recursive=True)

# 逐帧处理
for img in tqdm(img_list):
    frame = mmcv.imread('out_dirs/' + img)
    result = inference_detector(mask_rcnn_model, frame)

    mask = None
    masks = result[1][0]
    for i in range(len(masks)):
        if result[0][0][i][-1] >= 0.5:
            if not mask is None:
                mask = mask | masks[i]
            else:
                mask = masks[i]

    if mask is None:
        continue

    # 获取各通道mask
    masked_b = frame[:, :, 0] * mask
    masked_g = frame[:, :, 1] * mask
    masked_r = frame[:, :, 2] * mask
    # 合并各通道
    masked = np.concatenate([masked_b[:,:,None], masked_g[:,:,None], masked_r[:,:,None]],axis=2)

    # frame转灰度图
    un_mask = 1 - mask
    frame_b = frame[:, :, 0] * un_mask
    frame_g = frame[:, :, 1] * un_mask
    frame_r = frame[:, :, 2] * un_mask
    frame = np.concatenate([frame_b[:, :, None], frame_g[:, :, None], frame_r[:, :, None]], axis=2).astype(np.uint8)
    frame = mmcv.bgr2gray(frame, keepdim=True)
    frame = np.concatenate([frame, frame, frame], axis=2)
    # 合并
    frame += masked

    mmcv.imwrite(frame, 'splashed_dir/' + img)

print("detect done...")

# 生成视频
mmcv.frames2video('splashed_dir', 'out.mp4', fourcc='mp4v')