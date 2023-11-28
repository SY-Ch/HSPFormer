import os
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms
import cv2
from dataloader import custom_transforms as tr
import warnings

class KITTI360_Dataset(data.Dataset):
    NUM_CLASSES = 2

    CLASSES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 
                'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

    def __init__(self, cfg, root='', split='train'):

        self.ID2TRAINID = {0:255, 1:255, 2:255, 3:255, 4:255, 5:255, 6:255, 7:0, 8:1, 9:255, 10:255, 11:2, 12:3, 13:4, 14:255, 15:255, 16:255, 17:5, 18:255, 19:6, 
    20:7, 21:8, 22:9, 23:10, 24:11, 25:12, 26:13, 27:14, 28:15, 29:255, 30:255, 31:16, 32:17, 33:18, 34:2, 35:4, 36:255, 37:5, 38:255, 39:255, 40:255, 41:255, 42:255, 43:255, 44:255, -1:255}

        self.root = root
        self.split = split
        self.cfg = cfg
        self.images = {}
        self.disparities = {}
        self.labels = {}
        self.calibs = {}

        self.image_base = os.path.join(self.root, 'data_2d_raw', self.split)
        self.disparity_base = os.path.join(self.root, 'data_2d_depth', self.split)
        self.label_base = os.path.join(self.root, 'data_2d_semantics', self.split)

        self.images[split] = []
        self.images[split] = self.recursive_glob(rootdir=self.image_base, suffix='.png')
        self.images[split].sort()

        self.disparities[split] = []
        self.disparities[split] = self.recursive_glob(rootdir=self.disparity_base, suffix='.png')
        self.disparities[split].sort()

        self.labels[split] = []
        self.labels[split] = self.recursive_glob(rootdir=self.label_base, suffix='.png')
        self.labels[split].sort()

        self.label_map = np.arange(256)
        for id, trainid in self.ID2TRAINID.items():
            self.label_map[id] = trainid

        if not self.images[split]:
            raise Exception("No RGB images for split=[%s] found in %s" % (split, self.image_base))
        if not self.disparities[split]:
            raise Exception("No depth images for split=[%s] found in %s" % (split, self.disparity_base))

        print("Found %d %s RGB images" % (len(self.images[split]), split))
        print("Found %d %s disparity images" % (len(self.disparities[split]), split))

    def __len__(self):
        return len(self.images[self.split])

    def __getitem__(self, index):

        img_path = self.images[self.split][index].rstrip()
        disp_path = self.disparities[self.split][index].rstrip()
        lbl_path = self.labels[self.split][index].rstrip()
        
        label_image = cv2.imread(lbl_path,cv2.IMREAD_GRAYSCALE)
        oriHeight, oriWidth = label_image.shape

        label_image = self.label_map[label_image]

        _img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)

        disp_image = cv2.imread(disp_path, cv2.COLOR_BGR2RGB)
        

        sample = {'image': _img, 'depth': disp_image, 'label': label_image}

        self.path = img_path

        if self.split == 'train':
            sample = self.transform_tr(sample)
        elif self.split == 'val':
            sample = self.transform_val(sample)
        elif self.split == 'test':
            sample = self.transform_ts(sample)
        else:
            sample = self.transform_ts(sample)

        sample['img_path'] = img_path
        sample['depth_path'] = disp_path
        sample['oriHeight'] = oriHeight
        sample['oriWidth'] = oriWidth
        sample['oriSize'] = (oriHeight,oriWidth)

        sample['name'] = "/".join(img_path.rsplit("/", 2)[1:])


        return sample

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]


    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
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
            tr.Resize(size=(self.cfg["TRAIN"]["size"][0], self.cfg["TRAIN"]["size"][1])),
            tr.Normalize_tensor(rgb_mean=[0.485, 0.456, 0.406], rgb_std=[0.229, 0.224, 0.225]),
            ])
        return composed_transforms(sample)

    def transform_ts(self, sample):
        composed_transforms = transforms.Compose([
            tr.Resize(size=(self.cfg["TRAIN"]["size"][0], self.cfg["TRAIN"]["size"][1])),
            tr.Normalize_tensor(rgb_mean=[0.485, 0.456, 0.406], rgb_std=[0.229, 0.224, 0.225]),
            ])
        return composed_transforms(sample)