import numpy as np
import torch

def get_nyudepthv2_colors():
        
        return torch.tensor([[255, 194, 7],[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
               [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
               [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
               [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
               [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
               [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
               [224, 255, 8], [102, 8, 255], [255, 61, 6],[255,255,255] ])

def get_KIITI360_color():
      
      return torch.tensor([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], 
                [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32],[255,255,255]])

def get_nyudepthv2_classes():
        return ['otherprop','wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','blinds',
        'desk','shelves','curtain','dresser','pillow','mirror','floor mat','clothes','ceiling','books','refridgerator',
        'television','paper','towel','shower curtain','box','whiteboard','person','night stand','toilet',
        'sink','lamp','bathtub','bag','otherstructure','otherfurniture',]

def colorize_segmentation(segmentation_result, color_mapping):
    h, w = segmentation_result.shape
    colored_result = np.zeros((h, w, 3), dtype=np.uint8)

    for cls_idx, color in enumerate(color_mapping):
        colored_result[segmentation_result == cls_idx] = color

    return colored_result