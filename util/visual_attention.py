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
    """
    img_path: 读取图片的位置
    attention_mask: 2-D 的numpy矩阵
    ratio:  放大或缩小图片的比例，可选
    cmap:   attention map的style，可选
    """
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
    # # 将0值设置为透明色
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

def viual_attention(args, model, dataloader):
    print('start viual_attention!')
    with torch.no_grad():
        model.eval()
        # hist = np.zeros((args.num_classes, args.num_classes))
        for i, data in enumerate(dataloader):
            image, depth, label = data['image'], data['depth'], data['label']
            image_size = data['oriSize'][0]
            oriSize = (image_size[0].item(), image_size[1].item())
            model.set_input(data,image_size)
            model.forward()
            model.get_loss()
            output = model.output
            
            output = torch.nn.functional.interpolate(output,size=(oriSize[1],oriSize[0]), mode='bilinear', align_corners= True)
            
            
            print(model.get_image_names()[0].split('.')[0])

            # #attention2=[]
            # #attention3=[]
            # attention4=[]
            # fusion=[]
            # #DS_Combin=[]
            # hooks = [
            #     #model.module.attention2.mu.register_forward_hook(
            #     #    lambda self, input, output: attention2.append(output.cpu())
            #     #),
            #     #model.module.attention3.mu.register_forward_hook(
            #     #    lambda self, input, output: attention3.append(output.cpu())
            #     #),
            #     model.module.attention4.mu.register_forward_hook(
            #         lambda self, input, output: attention4.append(output.cpu())
            #     ),
            #     model.module.fusion.register_forward_hook(
            #         lambda self, input, output: fusion.append(output.cpu())
            #     #),
            #     #model.module.DS_Combin.register_forward_hook(
            #     #    lambda self, input, output: DS_Combin.append(output.cpu())
            #     )
            # ]
            # # get predict image
            # evidence, evidence_a, alpha, alpha_a, final = model(image, depth)
            # temp_name = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
            #    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
            #    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
            #    'bicycle']
            temp_name = ['unlabeled' , 'ego vehicle' , 'rectification border' , 'out of roi'  , 
                'static' , 'dynamic' , 'ground' , 'road' , 'sidewalk'  , 'parking' , 
                'rail track'  , 'building'  , 'wall' , 'fence'  , 'guard rail'  , 
                'bridge' , 'tunnel' , 'pole' , 'polegroup' , 'traffic light' , 
                'traffic sign'  , 'vegetation'  , 'terrain' , 'sky'  , 'person' , 
                'rider'  , 'car'  , 'truck'  , 'bus'  , 'caravan' , 'trailer' , 
                'train'  , 'motorcycle'  , 'bicycle' ]
            
            temp_list = []
            output = torch.squeeze(output)

            for i in range(len(temp_name)):
                class_i = torch.squeeze(output[i]).cpu().numpy()
                temp_list.append(class_i)

            
            print("!!!!!!!!!!!!!!")
            #temp = [fusion_1,fusion_2,attention3_1,attention3_2,attention4_,DS_Combin_1,DS_Combin_2,final_1,final_2,final_3,final_4]
            # temp_list = [am1_1,am1_2,am2_1,am2_2,am3_1,am3_2,am4_1,am4_2,
            #              fuse_am1_1,fuse_am1_2,fuse_am2_1,fuse_am2_2,fuse_am3_1,
            #              fuse_am3_2,fuse_am4_1,fuse_am4_2,fuse_1_1,fuse_1_2,fuse_2_1,fuse_2_2,fuse_3_1,fuse_3_2]
            # #temp_name = ["fusion_1","fusion_2","attention3_1","attention3_2","attention4_","DS_Combin_1","DS_Combin_2","final_1","final_2","final_3","final_4"]
            # temp_name = ["am1_1","am1_2","am2_1","am2_2","am3_1","am3_2","am4_1","am4_2",
            #              "fuse_am1_1","fuse_am1_2","fuse_am2_1","fuse_am2_2","fuse_am3_1",
            #              "fuse_am3_2","fuse_am4_1","fuse_am4_2","fuse_1_1","fuse_1_2","fuse_2_1","fuse_2_2","fuse_3_1","fuse_3_2"]

            img_path = data['img_path'][0]
            img_name = img_path.split('/')[-1]
            save_name = img_name.split('_')[0]+'_road_'+img_name.split('_')[1]
            # save_path = os.path.join(args.save_vis_path , "am1_1")
            # visulize_attention_ratio(img_path, am1_1, os.path.join(save_path, save_name))
            # show_feature_map(attention4_,save_name)
            for i in range(len(temp_list)):
                save_path = os.path.join(args.save_vis_path , temp_name[i])
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                visulize_attention_ratio(img_path, temp_list[i], os.path.join(save_path, save_name))
