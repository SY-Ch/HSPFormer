import torch
import os
import os
import importlib

import torch
import torch.nn as nn

from models.build_ufs import *
import torch.nn.functional as F
from dataloader.Nyu_depthV2 import NYU_depth_V2_Dataset
from dataloader.KITTI360 import KITTI360_Dataset


def make_dir(cfg):
    pth_save_dir = os.path.join(cfg["MODEL"]["model_names"], cfg["MODEL"]["model_spec"])

    if not cfg["continue_train"]:
        if cfg["DATASET"]["dataset"] == 'Cityscapes' or cfg["DATASET"]["dataset"] == 'cityscapes':
            cfg["DATASET"]["name"] = cfg["DATASET"]["name"] + "_" + str(cfg["TRAIN"]["size"][0]) + "x" + str(cfg["TRAIN"]["size"][1])
            
        if(os.path.exists(os.path.join(cfg["TRAIN"]["checkpoints_dir"] , pth_save_dir , cfg["DATASET"]["name"])) and (any(file.endswith('.ckpt') for file in os.listdir(os.path.join(cfg["TRAIN"]["checkpoints_dir"] , pth_save_dir , cfg["DATASET"]["name"]))))):
            name_index = 1
            while (os.path.exists(os.path.join(cfg["TRAIN"]["checkpoints_dir"] , pth_save_dir , cfg["DATASET"]["name"] + "_" + str(name_index))) 
                    and (any(file.endswith('.ckpt') for file in os.listdir(os.path.join(cfg["TRAIN"]["checkpoints_dir"] , pth_save_dir , cfg["DATASET"]["name"] + "_" + str(name_index)))))):
                    
                    name_index = name_index + 1

            cfg["DATASET"]["name"] = cfg["DATASET"]["name"] + "_" + str(name_index)

    return os.path.join(cfg["TRAIN"]["checkpoints_dir"], pth_save_dir, cfg["DATASET"]["name"])


def build_data(cfg):
    if cfg["DATASET"]["dataset"] == 'NYU_depth_V2':
        train_set = NYU_depth_V2_Dataset(cfg, root=cfg["DATASET"]["dataroot"], split='train')
        val_set = NYU_depth_V2_Dataset(cfg, root=cfg["DATASET"]["dataroot"], split='test')
    elif cfg["DATASET"]["dataset"] == 'KITTI360':
        train_set = KITTI360_Dataset(cfg, root=cfg["DATASET"]["dataroot"], split='train')
        val_set = KITTI360_Dataset(cfg, root=cfg["DATASET"]["dataroot"], split='val')
    
    return train_set,val_set

def build_model(train_opt, cfg):
    filters = [64, 128, 320, 512]

    model_filename = "models.encoders." + cfg["MODEL"]["model_names"]
    model_modellib = importlib.import_module(model_filename)
    encoder = getattr(model_modellib, cfg["MODEL"]["model_spec"])


    model = seg_network_ufs(encoder(), DecoderHead(in_channels=filters,num_classes=cfg["DATASET"]["num_labels"],norm_layer=nn.GroupNorm,embed_dim=filters[-1]), train_opt, cfg)

    return model

def get_ckpt_file(directory):
    print(directory)
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".ckpt"):
                print(os.path.join(root, file))
                return os.path.join(root, file)
    return None

def slide_inference(model, inputs, num_classes, img_size):
        """Inference by sliding-window with overlap. """

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        h_stride, w_stride = (768, 768)
        h_crop, w_crop = (768, 768)
        rgb_img = inputs['image']
        depth_img = inputs['depth']
        batch_size, _, h_img, w_img = rgb_img.size()
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = rgb_img.new_zeros((batch_size, num_classes, h_img, w_img)).to(device)
        count_mat = rgb_img.new_zeros((batch_size, 1, h_img, w_img)).to(device)
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_rgb = rgb_img[:, :, y1:y2, x1:x2]
                crop_depth = depth_img[:, :, y1:y2, x1:x2]
                inputs['image'] = crop_rgb
                inputs['depth'] = crop_depth
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                model.set_input(inputs, img_size)
                model.forward()
                crop_seg_logit = model.output
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        model.slide_output(seg_logits)

def whole_inference(model, inputs, oriSize):
    """Inference with full image. """

    model.set_input(inputs, oriSize)
    model.forward()