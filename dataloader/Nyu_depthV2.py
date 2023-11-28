import os
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms
import cv2
import glob
from dataloader import custom_transforms as tr
import matplotlib.pyplot as plt
from cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling import *

class NYU_depth_V2_Dataset(data.Dataset):
    NUM_CLASSES = 40
    
    CLASSES = ['wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','blinds',
    'desk','shelves','curtain','dresser','pillow','mirror','floor mat','clothes','ceiling','books','refridgerator',
    'television','paper','towel','shower curtain','box','whiteboard','person','night stand','toilet',
    'sink','lamp','bathtub','bag','otherstructure','otherfurniture','otherprop']

    def __init__(self, cfg, root='./data/KITTI/', split='training'):

        self.root = root
        self.split = split
        self.cfg = cfg
        self.images = {}
        self.depths = {}
        self.labels = {}
        self.calibs = {}

        self.image_base = os.path.join(self.root, self.split, 'image')
        self.depth_base = os.path.join(self.root, self.split, 'hha')
        self.label_base = os.path.join(self.root, self.split, 'label')

        self.images[split] = []
        self.images[split].extend(glob.glob(os.path.join(self.image_base, '*.jpg')))
        self.images[split].sort()

        self.depths[split] = []
        self.depths[split].extend(glob.glob(os.path.join(self.depth_base, '*.png')))
        self.depths[split].sort()

        self.labels[split] = []
        self.labels[split].extend(glob.glob(os.path.join(self.label_base, '*.png')))
        self.labels[split].sort()

        if not self.images[split]:
            raise Exception("No RGB images for split=[%s] found in %s" % (split, self.image_base))
        if not self.depths[split]:
            raise Exception("No depth images for split=[%s] found in %s" % (split, self.depth_base))

        print("Found %d %s RGB images" % (len(self.images[split]), split))
        print("Found %d %s depth images" % (len(self.depths[split]), split))

    def __len__(self):
        return len(self.images[self.split])

    def __getitem__(self, index):
        

        img_path = self.images[self.split][index].rstrip()
        depth_path = self.depths[self.split][index].rstrip()

        _img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)

        _depth = cv2.imread(depth_path, cv2.COLOR_BGR2RGB)

        lbl_path = self.labels[self.split][index].rstrip()

        label_image=cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
        
        oriHeight, oriWidth = label_image.shape

        label_image[label_image==255]=0
        label_image -= 1

        sample = {'image': _img, 'depth': _depth, 'label': label_image}

        if self.split == 'train':
            sample = self.transform_tr(sample)
        elif self.split == 'val':
            sample = self.transform_val(sample)
        elif self.split == 'test':
            sample = self.transform_ts(sample)
        else:
            sample = self.transform_ts(sample)


        sample['oriHeight'] = oriHeight
        sample['oriWidth'] = oriWidth

        sample['img_path'] = img_path
        sample['depth_path'] = depth_path
        sample['name'] = img_path.rsplit('/',1)[1]
        sample['oriSize'] = (oriHeight,oriWidth)

        return sample

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            # tr.Resize(size=(self.args.crop_width, self.args.crop_height)),
            tr.RandomFlip(),
            tr.RandomResize(ratio_range=[0.5, 2.0]),
            # tr.PhotoMetricDistortion(),
            tr.Normalize_tensor(rgb_mean=[0.485, 0.456, 0.406], rgb_std=[0.229, 0.224, 0.225]),
            tr.Padding(img_size=(self.cfg["TRAIN"]["size"][0], self.cfg["TRAIN"]["size"][1])),
            tr.RandomCrop(crop_size=(self.cfg["TRAIN"]["size"][0], self.cfg["TRAIN"]["size"][1])),
            tr.RandomColorJitter(p=0.2), # 
            tr.RandomGaussianBlur((3, 3), p=0.2), #
            ])
        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.Normalize_tensor(rgb_mean=[0.485, 0.456, 0.406], rgb_std=[0.229, 0.224, 0.225]),
            ])
            
        return composed_transforms(sample)

    def transform_ts(self, sample):
        composed_transforms = transforms.Compose([
            tr.Normalize_tensor(rgb_mean=[0.485, 0.456, 0.406], rgb_std=[0.229, 0.224, 0.225]),
            ])

        return composed_transforms(sample)
