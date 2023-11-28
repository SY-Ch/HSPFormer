import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from argparse import ArgumentParser
import numpy as np

from util.util import *
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import LearningRateFinder

import yaml

parser = ArgumentParser()

parser.add_argument('--cfg', type=str, default='configs/KITTI360.yaml', help='Configuration file to use')

train_opt = parser.parse_args()

with open(train_opt.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

train_opt.isTrain=True

train_opt.save_pth_dir = make_dir(cfg)
_, val_set = build_data(cfg)

data_loader_val = DataLoader(val_set, batch_size = 1,
                                num_workers=cfg["TRAIN"]["num_workers"],
                                pin_memory=True)

checkpoint_callback = ModelCheckpoint(
     monitor='index/IOU',
     dirpath=train_opt.save_pth_dir,
     filename=cfg["DATASET"]["name"],
     mode= 'max',
 )

lr_monitor = LearningRateMonitor(logging_interval='epoch')

tensorboard_logger = TensorBoardLogger(save_dir=train_opt.save_pth_dir)

train_opt.total_samples = len(data_loader_val)

model = build_model(train_opt, cfg)

trainer = pl.Trainer(strategy=DDPStrategy(),devices=cfg["TRAIN"]["node"], max_epochs=cfg["TRAIN"]["nepoch"], callbacks = [checkpoint_callback,lr_monitor], default_root_dir= train_opt.save_pth_dir, logger=tensorboard_logger, log_every_n_steps= 50,check_val_every_n_epoch = cfg["TRAIN"]["eval_interval"])

pth_dir = get_ckpt_file(train_opt.save_pth_dir)

state_dict = torch.load(pth_dir, map_location="cpu")
model.load_state_dict(state_dict, strict=True)

trainer = pl.Trainer(devices=train_opt.node, default_root_dir= train_opt.save_pth_dir, logger=tensorboard_logger,inference_mode=False)

trainer.test(model=model, dataloaders=data_loader_val)
