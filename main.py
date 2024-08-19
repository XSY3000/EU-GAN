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

config = Config('configs/config1.yml')
trainer = Trainer(config)

# trainer.train()

trainer.test(weight='weights/EU-GAN/bestmodel.pth')

# image_path = 'your/image/path'
# save_path = 'your/save/path'  # if '', the result will be saved in the log directory.
# trainer.infer(image_path, save_path, weights='weights/EU-GAN/bestmodel.pth')
