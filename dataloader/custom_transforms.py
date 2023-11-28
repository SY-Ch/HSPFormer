import torch
import random
import numpy as np
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance, ImageChops
import cv2
import torch.nn.functional as F
import torchvision.transforms.functional as TF 

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, rgb_mean=(0., 0., 0.), rgb_std=(1., 1., 1.)):
        self.rgb_mean = np.array(rgb_mean)
        self.rgb_std = np.array(rgb_std)

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']

        mean = np.float64(self.rgb_mean.reshape(1, -1))
        stdinv = 1 / np.float64(self.rgb_std.reshape(1, -1))
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
        cv2.subtract(img, mean, img)  # inplace
        cv2.multiply(img, stdinv, img)  # inplace

        # Convert the image data type to float32 for division
        # depth = depth.astype(np.float32)

        # # Divide each pixel value by 255 to normalize the image
        # depth = cv2.divide(depth, 255.0)
        # img = img.astype(np.float32)
        # depth = depth.astype(np.float32)

        # img -= self.rgb_mean
        # img /= self.rgb_std

        # depth /= 255.0

        return {'image': img,
                'depth': depth,
                'label': mask}
    
class Normalize_tensor(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, rgb_mean=(0., 0., 0.), rgb_std=(1., 1., 1.), have_depth = True):
        self.rgb_mean = np.array(rgb_mean)
        self.rgb_std = np.array(rgb_std)
        self.have_depth = have_depth

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']

        img = np.array(img)

        img = img.astype(np.float64) / 255.0
        img = img - self.rgb_mean
        img = img / self.rgb_std

        img = img.transpose(2, 0, 1)

        if self.have_depth:
            depth = np.array(depth)

            depth = depth.astype(np.float64) / 255.0
            depth = depth - self.rgb_mean
            depth = depth / self.rgb_std

            depth = depth.transpose(2, 0, 1)

        img = torch.FloatTensor(img)
        depth = torch.FloatTensor(depth)
        mask = torch.LongTensor(mask)

        return {'image': img,
                'depth': depth,
                'label': mask}


class ToTensor(object):
    """Convert Image object in sample to Tensors."""
    
    def __init__(self, depth=False ):
        self.depth = depth


    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        if self.depth:
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)


        img = torch.tensor(img, dtype=torch.float32)
        depth = torch.tensor(depth, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.long)

        if len(depth.shape)== 2:
            depth = depth.unsqueeze(0)

        return {'image': img,
                'depth': depth,
                'label': mask}


class ColorJitter(object):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.brightness = [max(1 - brightness, 0), 1 + brightness]
        self.contrast = [max(1 - contrast, 0), 1 + contrast]
        self.saturation = [max(1 - saturation, 0), 1 + saturation]

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        img = ImageEnhance.Brightness(img).enhance(r_brightness)
        img = ImageEnhance.Contrast(img).enhance(r_contrast)
        img = ImageEnhance.Color(img).enhance(r_saturation)
        return {'image': img,
                'depth': depth,
                'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'depth': depth,
                'label': mask}


class HorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'depth': depth,
                'label': mask}


class RandomGaussianBlur(object):
    def __init__(self, radius=1):
        self.radius = radius

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=self.radius*random.random()))

        return {'image': img,
                'depth': depth,
                'label': mask}


class RandomGaussianNoise(object):
    def __init__(self, mean=0, sigma=10):
        self.mean = mean
        self.sigma = sigma

    def gaussianNoisy(self, im, mean=0, sigma=10):
        noise = np.random.normal(mean, sigma, len(im))
        im = im + noise

        im = np.clip(im, 0, 255)
        return im

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']
        if random.random() < 0.5:
            img = np.asarray(img)
            img = img.astype(np.int)
            width, height = img.shape[:2]
            img_r = self.gaussianNoisy(img[:, :, 0].flatten(), self.mean, self.sigma)
            img_g = self.gaussianNoisy(img[:, :, 1].flatten(), self.mean, self.sigma)
            img_b = self.gaussianNoisy(img[:, :, 2].flatten(), self.mean, self.sigma)
            img[:, :, 0] = img_r.reshape([width, height])
            img[:, :, 1] = img_g.reshape([width, height])
            img[:, :, 2] = img_b.reshape([width, height])
            img = Image.fromarray(np.uint8(img))
        return {'image': img,
                'depth': depth,
                'label': mask}
    
    
class PhotoMetricDistortion(object):

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if random.randint(0, 1) == 1:
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if random.randint(0, 1) == 1:
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if random.randint(0, 1) == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def hue(self, img):
        """Hue distortion."""
        if random.randint(0, 1) == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :,
                0] = (img[:, :, 0].astype(int) +
                      random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def __call__(self, sample):

        img = sample['image']
        depth = sample['depth']
        mask = sample['label']
        # random brightness
        img = self.brightness(img)

        mode = random.randint(0, 1)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        return {'image': img,
                'depth': depth,
                'label': mask}
    


class RandomFlip(object):
    def __init__(self,prob=0.5):
        self.prob =prob

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']
        if random.random() < self.prob:
            img = cv2.flip(img, 1)
            depth = cv2.flip(depth, 1)
            mask = cv2.flip(mask, 1)

        return {'image': img,
                'depth': depth,
                'label': mask}



class Resize(object):
    """Resize rgb and label images, while keep depth image unchanged. """
    def __init__(self, size):
        self.size = size    # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']

        # resize rgb and label
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, self.size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        # mask = np.array(mask)
        # mask = resize_no_new_pixel(mask,self.size[0],self.size[1])
        # print(img.size)
        # print(depth.size)
        # print(mask.size)

        return {'image': img,
                'depth': depth,
                'label': mask}
    

class Resize_RGB(object):
    """Resize rgb and label images, while keep depth image unchanged. """
    def __init__(self, size):
        self.size = size    # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']

        # resize rgb and label
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
        # mask = np.array(mask)
        # mask = resize_no_new_pixel(mask,self.size[0],self.size[1])
        # print(img.size)
        # print(depth.size)
        # print(mask.size)

        return {'image': img,
                'depth': depth,
                'label': mask}
    

class RandomResize(object):
    """Randomly resize RGB and label images while keeping the depth image unchanged."""

    def __init__(self, ratio_range):
        self.ratio_range = ratio_range  # size_range: (min_size, max_size)

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']

        assert img.shape[:2] == depth.shape[:2]

        img_scale = img.shape[:2]  # (height, width)

        min_ratio, max_ratio = self.ratio_range
        ratio = random.uniform(min_ratio, max_ratio)

        # 计算新的尺寸
        new_height = int(img_scale[0] * ratio)
        new_width = int(img_scale[1] * ratio)

        # Resize RGB and label images using OpenCV
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        return {
            'image': img,
            'depth': depth,
            'label': mask}
    

class Padding(object):
    """Randomly resize RGB and label images while keeping the depth image unchanged."""

    def __init__(self, img_size):
        self.img_size = img_size  # size_range: (min_size, max_size)

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']

        height,width  = mask.shape

        if height < self.img_size[0] or  width< self.img_size[1]:
        # 计算需要进行填充的尺寸
            # Calculate the required padding size
            pad_width = max(self.img_size[1] - width, 0)
            pad_height = max(self.img_size[0] - height, 0)
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top

            # 在图像周围进行填充
            img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom),value=0)
            depth = F.pad(depth, (pad_left, pad_right, pad_top, pad_bottom),value=0)
            mask = F.pad(mask, (pad_left, pad_right, pad_top, pad_bottom),value=255)
        

        return {
            'image': img,
            'depth': depth,
            'label': mask}

class RandomColorJitter(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):

        img = sample['image']
        depth = sample['depth']
        mask = sample['label']

        if random.random() < self.p:
            self.brightness = random.uniform(0.5, 1.5)
            img = TF.adjust_brightness(img, self.brightness)
            self.contrast = random.uniform(0.5, 1.5)
            img = TF.adjust_contrast(img, self.contrast)
            self.saturation = random.uniform(0.5, 1.5)
            img = TF.adjust_saturation(img, self.saturation)

        return {
            'image': img,
            'depth': depth,
            'label': mask}
    
class RandomGaussianBlur(object):
    def __init__(self, kernel_size: int = 3, p: float = 0.5):
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, sample):

        img = sample['image']
        depth = sample['depth']
        mask = sample['label']

        if random.random() < self.p:
            img = TF.gaussian_blur(img, self.kernel_size)
            # img = TF.gaussian_blur(img, self.kernel_size)

        return {
            'image': img,
            'depth': depth,
            'label': mask}

    

class RandomCrop(object):
    """Resize rgb and label images, while keep depth image unchanged. """
    def __init__(self, crop_size):
        self.crop_size = crop_size    # size: (h, w)

    def random_crop_image_gt_depth(self, image, gt, depth):
    # 进行随机裁剪
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.crop_size)
        cropped_image = transforms.functional.crop(image, i, j, h, w)
        cropped_gt = transforms.functional.crop(gt, i, j, h, w)
        cropped_depth = transforms.functional.crop(depth, i, j, h, w)
        return cropped_image, cropped_gt, cropped_depth

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']

        cropped_image, cropped_gt, cropped_depth = self.random_crop_image_gt_depth(img, mask, depth)

        return {'image': cropped_image,
                'depth': cropped_depth,
                'label': cropped_gt}

class MyGaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"

    def __init__(self, radius=2, bounds=None):
        self.radius = radius
        self.bounds = bounds

    def filter(self, image):
        if self.bounds:
            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)

class USM_sharpen(object):
    def __init__(self, radius=7.1):
        self.radius = radius

    def __call__(self, sample):
        img = sample['image']
        depth1 = sample['depth']
        depth2 = sample['depth']
        mask = sample['label']      

        depth2 = depth2.filter(MyGaussianBlur(radius=7.1))

        depth2=ImageChops.add(depth2,depth2,9)
        depth1=ImageChops.add(depth1,depth1,8)
        depth=ImageChops.subtract(depth2,depth1,1)
        # depth=(depth1-0.9*depth_)/0.1

        save_path="/home/ht/csy/master/test_vi/"+sample['name']

        depth.save(save_path)

        return {'image': img,
                'depth': depth,
                'label': mask}

def resize_no_new_pixel(src_img,out_h,out_w):
    dst_img = np.zeros((out_h,out_w))
    

    height =  src_img.shape[0]
    width  =  src_img.shape[1]

    w_scale = float(width/out_w)
    h_scale = float(height/out_h)
    
    for j in range(out_h):
        for i in range(out_w):
            raw_w = int(i*w_scale)
            raw_h = int(j*h_scale)
            dst_img[j][i]=src_img[raw_h][raw_w]

    return dst_img

def pad_and_resize_image(image, size):
    target_w, target_h = size[0],size[1]
    # 获取原始图像的尺寸
    original_w, original_h = image.size

    # 计算需要填充的宽度和高度
    pad_w = target_w - original_w
    pad_h = target_h - original_h

    # 计算填充的左、右、上、下边距
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    # 使用白色（255, 255, 255）填充图像
    padded_image = Image.new(image.mode, (target_w, target_h), 255)
    padded_image.paste(image, (left, top))

    return padded_image