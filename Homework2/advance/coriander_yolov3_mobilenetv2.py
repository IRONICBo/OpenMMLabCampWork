checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=192)
model = dict(
    type='YOLOV3',
    backbone=dict(
        type='MobileNetV2',
        out_indices=(2, 4, 6),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://mmdet/mobilenet_v2')),
    neck=dict(
        type='YOLOV3Neck',
        num_scales=3,
        in_channels=[320, 96, 32],
        out_channels=[96, 96, 96]),
    bbox_head=dict(
        type='YOLOV3Head',
        # num_classes=80,
        num_classes=1,
        in_channels=[96, 96, 96],
        out_channels=[96, 96, 96],
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(220, 125), (128, 222), (264, 266)],
                        [(35, 87), (102, 96), (60, 170)],
                        [(10, 15), (24, 36), (72, 42)]],
            strides=[32, 16, 8]),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[32, 16, 8],
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_conf=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_xy=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=2.0,
            reduction='sum'),
        loss_wh=dict(type='MSELoss', loss_weight=2.0, reduction='sum')),
    train_cfg=dict(
        assigner=dict(
            type='GridAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        conf_thr=0.005,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=100))
# 修改为Voc
dataset_type = 'VOCDataset'
data_root = 'data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=[123.675, 116.28, 103.53],
        to_rgb=True,
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(320, 320), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(320, 320),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2, # Batch size of a single GPU 单个GPU的batch
    workers_per_gpu=2, # Worker to pre-fetch data for each single GPU 预取数据的batch
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file=[
                data_root + 'VOC2007/ImageSets/Main/trainval.txt',
            ],
            img_prefix=[data_root + "VOC2007/"],
            pipeline=train_pipeline)),
    val=dict(
        # 这个需要从路径中推断年份，所以需要加上年份的目录
        # can not infer year
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + "VOC2007/",
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + "VOC2007/",
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='mAP')
optimizer = dict(type='SGD', lr=0.003, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=4000,
    warmup_ratio=0.0001,
    step=[24, 28])
runner = dict(type='EpochBasedRunner', max_epochs=30)

# evaluation = dict(interval=1, metric=['bbox'])
# 使用VOC的数据集只支持mAP和recall，不支持bbox
evaluation = dict(interval=1, metric=['mAP'])
find_unused_parameters = True
work_dir = 'work/coriander_yolov3_mobilenetv2'
gpu_ids = [0]

# 训练的代码
# python ./tools/train.py coriander_yolov3_mobilenetv2.py

# difficult有问题

# TypeError: Caught TypeError in DataLoader worker process 0.
# Original Traceback (most recent call last):
#   File "/home/asklv/Tools/anaconda/envs/open-mmlab/lib/python3.7/site-packages/torch/utils/data/_utils/worker.py", line 302, in _worker_loop
#     data = fetcher.fetch(index)
#   File "/home/asklv/Tools/anaconda/envs/open-mmlab/lib/python3.7/site-packages/torch/utils/data/_utils/fetch.py", line 49, in fetch
#     data = [self.dataset[idx] for idx in possibly_batched_index]
#   File "/home/asklv/Tools/anaconda/envs/open-mmlab/lib/python3.7/site-packages/torch/utils/data/_utils/fetch.py", line 49, in <listcomp>
#     data = [self.dataset[idx] for idx in possibly_batched_index]
#   File "/home/asklv/Projects/OpenMMLab/mmdetection/mmdet/datasets/dataset_wrappers.py", line 178, in __getitem__
#     return self.dataset[idx % self._ori_len]
#   File "/home/asklv/Tools/anaconda/envs/open-mmlab/lib/python3.7/site-packages/torch/utils/data/dataset.py", line 235, in __getitem__
#     return self.datasets[dataset_idx][sample_idx]
#   File "/home/asklv/Projects/OpenMMLab/mmdetection/mmdet/datasets/custom.py", line 220, in __getitem__
#     data = self.prepare_train_img(idx)
#   File "/home/asklv/Projects/OpenMMLab/mmdetection/mmdet/datasets/custom.py", line 238, in prepare_train_img
#     ann_info = self.get_ann_info(idx)

#   File "/home/asklv/Projects/OpenMMLab/mmdetection/mmdet/datasets/xml_style.py", line 114, in get_ann_info
#     difficult = 0 if difficult is None else int(difficult.text)
# TypeError: int() argument must be a string, a bytes-like object or a number, not 'NoneType'