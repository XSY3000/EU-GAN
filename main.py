import random

import numpy as np
import torch

from load_config import Config
from trainer import Trainer

# 固定随机种子
SEED = 1999
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# dataset_path = ['filedata/water-base', 'filedata/cotton']
config = Config('configs/config1.yml')
trainer = Trainer(config)
#trainer.train()
# trainer.eval()
#trainer.test('')
trainer.test('', weight='logs/2024-03-15-16-53-13-watercotton150k/checkpoints/generatorG_137500.pth')
#trainer.test('', weight='logs/2024-07-23-12-56-23-noEAM/checkpoints/generatorG_138000.pth')
# trainer.infer(r'testimg',
#               None,
#               weights='logs/2024-03-15-16-53-13-watercotton150k/checkpoints/generatorG_137500.pth')
# trainer.infer(image_path=r'F:\file\myRice\smallRootBoxData\first\warp_sample2\2_10_6',
#               weights='logs/2023-11-30-15-09-27-cotton0610/checkpoints/generatorG_71.pth')
# trainer.infer(image_path=r'filedata/riceroot/images', weights='logs/2024-03-15-16-53-13-watercotton150k/checkpoints/generatorG_137500.pth')
