import os
import time
from util.util import *
import numpy as np
import random
import torch
import cv2
import argparse
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image


def visulize_attention_ratio(img_path, attention_mask, save_path, ratio=1, cmap="jet"):

    print("load image from: ", img_path)
    # load the image
    img = Image.open(img_path, mode='r')
    img_h, img_w = img.size[0], img.size[1]
    plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))

    # scale the image
    img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
    img = img.resize((img_h, img_w))
    plt.imshow(img, alpha=1)
    plt.axis('off')
    
    # normalize the attention mask
    mask = cv2.resize(attention_mask, (img_h, img_w))
    normed_mask = mask / mask.max()
    normed_mask = (normed_mask * 255).astype('uint8')

    # colors = [(1, 0, 0, 1-i) for i in np.linspace(0, 1, 100)]
    # cmap_red_transparent = matplotlib.colors.LinearSegmentedColormap.from_list('mycmap', colors, N=100)

    # cmap = matplotlib.colors.ListedColormap(['none','lime' ])

    # cmap.set_bad(alpha=0)

    plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap=cmap)
    print("save image to: ", save_path)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(save_path)
    plt.close()

def visualize_attention_only(attention_mask, save_path, output_width=None, output_height=None, ratio=1, cmap="jet"):
    """
    attention_mask: 2-D numpy array
    output_width: output width of the image, optional
    output_height: output height of the image, optional
    ratio: resize ratio of the image, optional
    cmap: color map style of the attention map, optional
    """
    # Normalize the attention mask
    normed_mask = attention_mask / attention_mask.max()
    normed_mask = (normed_mask * 255).astype('uint8')
    
    # Determine the output size
    if output_width is not None and output_height is not None:
        resized_h, resized_w = output_height, output_width
    else:
        h, w = attention_mask.shape
        resized_h, resized_w = int(h * ratio), int(w * ratio)
    
    # Resize the attention mask
    resized_mask = cv2.resize(normed_mask, (resized_w, resized_h))
    
    plt.figure(figsize=(resized_w/100, resized_h/100))  # figsize in inches
    
    # Displaying the attention mask
    plt.imshow(resized_mask, interpolation='nearest', cmap=cmap)
    plt.axis('off')
    
    # Save the figure and release resources
    print("save image to: ", save_path)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def show_feature_map(feature_map,save_path):
    feature_map = feature_map.squeeze(0)
    feature_map = feature_map.cpu().numpy()
    plt.figure()
    plt.imshow(feature_map, cmap='gray')
    plt.axis('off')
    plt.savefig(save_path)
