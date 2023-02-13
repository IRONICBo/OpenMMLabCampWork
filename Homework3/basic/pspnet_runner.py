from mmengine.runner import Runner
from mmseg.utils import register_all_modules
from mmengine import Config

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

# 数据集图片和标注路径
data_root = 'data/tutorial'
img_dir = 'images'
ann_dir = 'masks'

# 类别和对应的颜色
classes = ('background', 'glomeruili')
palette = [[128, 128, 128], [151, 189, 8]]

@DATASETS.register_module()
class StanfordBackgroundDataset(BaseSegDataset):
  METAINFO = dict(classes = classes, palette = palette)
  def __init__(self, **kwargs):
    super().__init__(img_suffix='.png', seg_map_suffix='.png', **kwargs)

cfg = Config.fromfile('new_cfg.py')

# register all modules in mmseg into the registries
# do not init the default scope here because it will be init in the runner
register_all_modules(init_default_scope=False)
runner = Runner.from_cfg(cfg)

runner.train()