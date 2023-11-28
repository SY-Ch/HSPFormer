import torch
import torch.nn as nn

from models.decoders.MLPDecoder import DecoderHead
import pytorch_lightning as pl
import torch.nn.functional as F
from util.scheduler import *
from util.lovasz_losses import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from timm.scheduler import CosineLRScheduler, PolyLRScheduler
import math
import torchmetrics
from util.metrics import *
from util.loss import *
from util.visual_attention import visulize_attention_ratio,visualize_attention_only
from tabulate import tabulate
from pathlib import Path
import timm
import cv2

import torch.linalg as LA

from util.color_name import *

class seg_network_ufs(pl.LightningModule):
    """Our RoadSeg takes rgb and another (depth or normal) as input,
    and outputs freespace predictions.
    """

    def __init__(self, encoder, decoder,opt, cfg):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.opt = opt
        self.cfg = cfg
        # self.loss = OhemCrossEntropy(ignore_label = self.opt.ignore_index)
        self.depth_loss = nn.SmoothL1Loss()

        if len(cfg["MODEL"]["load_Pretraining"].strip()) != 0 :
            print('loading the model from %s' % cfg["MODEL"]["load_Pretraining"])
            state_dict = torch.load(cfg["MODEL"]["load_Pretraining"], map_location="cpu")
            self.encoder.load_state_dict(state_dict, strict=False)
        
    def training_step(self, batch, batch_idx):
        
        x = batch
        z = self.encoder(x['image'], x['depth'])
        x_hat = self.decoder(z)
        x_hat = F.interpolate(x_hat,size=(self.cfg["TRAIN"]["size"][0], self.cfg["TRAIN"]["size"][1]),mode='bilinear',align_corners=False)
        loss = F.cross_entropy(x_hat, x['label'], ignore_index = self.cfg["DATASET"]["ignore_index"])
        self.log("Loss/train_segloss", loss, on_step=True,on_epoch=True, sync_dist=True, batch_size= self.cfg["TRAIN"]["batch_size"], prog_bar=True)

        return loss
    
    def on_validation_epoch_start(self):
        self.metrics = Metrics(self.cfg["DATASET"]["num_labels"], self.cfg["DATASET"]["ignore_index"]).cuda()
        self.eval_last_path = os.path.join(self.opt.save_pth_dir , 'eval_last_{}.txt'.format(self.cfg["DATASET"]["dataset"]))
        with open(self.eval_last_path, 'a') as f:
            f.write("============== start evaluate============== ")
    
    def validation_step(self, batch, batch_idx):
        x = batch
        _, _, H, W = x['image'].shape
        z = self.encoder(x['image'], x['depth'])
        x_hat = self.decoder(z)
        x_hat = F.interpolate(x_hat,size=(H, W),mode='bilinear',align_corners=False)
        loss = F.cross_entropy(x_hat, x['label'], ignore_index = self.cfg["DATASET"]["ignore_index"])
        self.log("Loss/validation_loss", loss, on_step=True,on_epoch=True, sync_dist=True, batch_size= 1)
        x_hat = x_hat.softmax(dim=1)
        self.metrics.update(x_hat, x['label'])
        return loss

    
    def on_validation_epoch_end(self):
        sem_index = self.metrics.compute()
        self.log("index/IOU", sem_index['mIOU'], sync_dist=True, batch_size= 1,prog_bar=True)
        self.log("index/Accuracy", sem_index['mACC'], sync_dist=True, batch_size= 1)
        self.log("index/F1score", sem_index['mF1'], sync_dist=True, batch_size= 1)

        table = {
                'Class': list(self.opt.class_list) + ['Mean'],
                'IoU': sem_index['IOUs'] + [sem_index['mIOU']],
                'F1': sem_index['F1'] + [sem_index['mF1']],
                'Acc': sem_index['ACC'] + [sem_index['mACC']]
                }
        
        

        with open(self.eval_last_path, 'a') as f:
            f.write("\n============== Eval on {} {} images =================\n".format(self.cfg["MODEL"]["model_names"], self.cfg["DATASET"]["dataset"]))
            f.write("\n")
            print(tabulate(table, headers='keys'), file=f)


    def on_test_epoch_start(self):
        self.metrics = Metrics(self.cfg["DATASET"]["num_labels"], self.cfg["DATASET"]["ignore_index"]).cuda()
        self.eval_last_path = os.path.join(self.opt.save_pth_dir , 'eval_last_{}.txt'.format(self.cfg["DATASET"]["dataset"]))
        with open(self.eval_last_path, 'a') as f:
            f.write("============== start evaluate============== ")

    def test_step(self, batch, batch_idx):

        x = batch
        _, _, H, W = x['image'].shape
        z = self.encoder(x['image'],x['depth'])
        x_hat = self.decoder(z)
        x_hat = F.interpolate(x_hat,size=(H, W),mode='bilinear',align_corners=False)

        x_hat = x_hat.softmax(dim=1)

        self.metrics.update(x_hat, x['label'])

        x_hat = x_hat.argmax(dim=1)

        color = get_KIITI360_color()
        # color = get_nyudepthv2_colors()
        x_hat = x_hat.squeeze().cpu().to(int)
        x_hat[x_hat == 255] = 19 
        # x_hat[x_hat == 255] = 40 
        x_hat = color[x_hat].squeeze().to(torch.uint8).numpy()
        x_hat = x_hat

        visual_path = os.path.join(self.opt.save_pth_dir,"pred")
        Path(visual_path).mkdir(parents=True, exist_ok=True)

        cv2.imwrite(os.path.join(visual_path, batch['name'][0]),x_hat)

        return x_hat

    def on_test_epoch_end(self):
        index = self.metrics.compute()

        table = {
                'Class': list(self.opt.class_list) + ['Mean'],
                'IoU': index['IOUs'] + [index['mIOU']],
                'F1': index['F1'] + [index['mF1']],
                'Acc': index['ACC'] + [index['mACC']]
                }
        
        

        with open(self.eval_last_path, 'a') as f:
            f.write("\n============== Eval on {} {} images =================\n".format(self.cfg["MODEL"]["model_names"], self.cfg["DATASET"]["dataset"]))
            f.write("\n")
            print(tabulate(table, headers='keys'), file=f)
    
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg["OPTIMIZER"]["lr"], betas = (0.9, 0.999), weight_decay = self.cfg["OPTIMIZER"]["weight_decay"])

        scheduler = WarmupCosineAnnealingLR(optimizer,self.cfg["TRAIN"]["nepoch"],self.cfg["SCHEDULER"]["warmup_epoch"], math.ceil(self.opt.total_samples / int(self.cfg["TRAIN"]["node"])),self.cfg["SCHEDULER"]["lr_warmup"],self.cfg["SCHEDULER"]["warmup_ratio"])

        lr_scheduler_config = {
            "scheduler" : scheduler ,
            "interval" : "step" ,
            "frequency" : 1,
            }

        return dict(optimizer=optimizer, lr_scheduler=lr_scheduler_config)

