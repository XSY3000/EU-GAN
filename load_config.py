import os.path
from datetime import datetime
from shutil import copyfile

import yaml
from tensorboardX import SummaryWriter


class Config():
    def __init__(self, config_path):
        self.config_path = config_path
        self.load_config()

        self.logdir = os.path.join('logs', f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}-{self.proj_NAME}')
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        self.tensorboard = SummaryWriter(log_dir=os.path.join(self.logdir))
        self.logger = printlogger(self.logdir)
        self.save_config()

        self.l1lossWeight = self.loss_weights['L1Loss']
        self.BCELossWeight = self.loss_weights['BCELoss']
        self.DiceLossWeight = self.loss_weights['DiceLoss']
        self.PerceptualLossWeight = self.loss_weights['PerceptualLoss']

    def load_config(self):
        with open(self.config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        for key, value in config.items():
            setattr(self, key, value)

    def save_config(self):
        copyfile(self.config_path, os.path.join(self.logdir, f'{self.proj_NAME}.yml'))
        copyfile('trainer.py', os.path.join(self.logdir, 'trainer.py'))


class printlogger():
    def __init__(self, logdir):
        self.logdir = os.path.join(logdir, 'log.txt')

    def print(self, txt):
        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.logdir, 'a') as f:
            print(time, file=f, end='\t')
            print(txt, file=f)
            print(time, end='\t')
            print(txt)
