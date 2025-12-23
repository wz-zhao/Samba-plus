import cv2
import torch
import numpy as np

class Compose(object):
    def __init__(self, *ops):
        self.ops = ops

    def __call__(self, rgb,t,d, mask):
        for op in self.ops:
            rgb,t,d, mask = op(rgb,t,d, mask)
        return rgb,t,d, mask

class Normalize(object):
    def __init__(self, mean1,mean2,mean3, std1,std2,std3):
        self.mean1 = mean1
        self.mean2 = mean2
        self.mean3 = mean3
        self.std1 = std1
        self.std2 = std2
        self.std3 = std3

    def __call__(self, rgb,t,d, mask):
        rgb =  np.array(rgb).astype(np.float32) / 255.0
        rgb = (rgb - self.mean1)/self.std1
        t /= 255.0
        t = (t - self.mean2) / self.std2
        d /= 255.0
        d = (d - self.mean3) / self.std3
        mask /= 255
        # print(torch.max(torch.as_tensor(rgb)))
        return rgb,t,d, mask

class Minusmean(object):
    def __init__(self, mean1,mean2):
        self.mean1 = mean1
        self.mean2 = mean2

    def __call__(self, rgb,t,d, mask):
        rgb = rgb - self.mean1
        t = t - self.mean2
        d = d - self.mean1
        mask /= 255
        return rgb,t,d, mask


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, rgb,t,d, mask):
        rgb = cv2.resize(rgb, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        t = cv2.resize(t, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        d = cv2.resize(d, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize( mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return rgb, t, d, mask

class RandomCrop(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, rgb,t,d, mask):
        H,W,_ = rgb.shape
        xmin  = np.random.randint(W-self.W+1)
        ymin  = np.random.randint(H-self.H+1)
        rgb = rgb[ymin:ymin+self.H, xmin:xmin+self.W, :]
        t = t[ymin:ymin + self.H, xmin:xmin + self.W, :]
        d = d[ymin:ymin + self.H, xmin:xmin + self.W, :]
        mask = mask[ymin:ymin+self.H, xmin:xmin+self.W, :]
        return rgb, t, d, mask

class Random_rotate(object):
    def __call__(self, rgb,t,d, mask):
        angle = np.random.randint(-25,25)
        h,w,_ = rgb.shape
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(rgb, M, (w, h)), cv2.warpAffine(t, M, (w, h)), cv2.warpAffine(d, M, (w, h)), cv2.warpAffine(mask, M, (w, h))

class RandomHorizontalFlip(object):
    def __call__(self, rgb,t,d, mask):
        if np.random.randint(2)==1:
            rgb = rgb[:,::-1,:].copy()
            t = t[:, ::-1, :].copy()
            d = d[:, ::-1, :].copy()
            mask = mask[:,::-1,:].copy()
        return rgb,t,d, mask

class ToTensor(object):
    def __call__(self, rgb,t,d, mask):
        rgb = torch.from_numpy(rgb)
        rgb = rgb.permute(2, 0, 1)
        t = torch.from_numpy(t)
        t = t.permute(2, 0, 1)
        d = torch.from_numpy(d)
        d = d.permute(2, 0, 1)
        mask = torch.from_numpy(mask)
        mask = mask.permute(2, 0, 1)
        return rgb,t,d,mask.mean(dim=0, keepdim=True)
    
from PIL import Image, ImageOps, ImageEnhance
import random
class ColorEnhance(object):
    def __call__(self, rgb, t, d, mask):
        # 转 PIL 格式，只增强 rgb
        rgb_pil = Image.fromarray(rgb.astype(np.uint8))

        bright_intensity = random.randint(5, 15) / 10.0
        rgb_pil = ImageEnhance.Brightness(rgb_pil).enhance(bright_intensity)

        contrast_intensity = random.randint(5, 15) / 10.0
        rgb_pil = ImageEnhance.Contrast(rgb_pil).enhance(contrast_intensity)

        color_intensity = random.randint(0, 20) / 10.0
        rgb_pil = ImageEnhance.Color(rgb_pil).enhance(color_intensity)

        sharp_intensity = random.randint(0, 30) / 10.0
        rgb_pil = ImageEnhance.Sharpness(rgb_pil).enhance(sharp_intensity)

        # 转回 numpy
        rgb = np.array(rgb_pil)
        return rgb, t, d, mask


class SaltPepperNoise(object):
    """给 rgb 图像添加椒盐噪声"""
    def __init__(self, prob=0.02):
        """
        prob: 噪声比例，越大噪声越强
        """
        self.prob = prob

    def __call__(self, rgb, t, d, mask):
        h, w, c = rgb.shape
        num_noise = int(h * w * self.prob)
        for _ in range(num_noise):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            rgb[y, x, np.random.randint(0, c)] = np.random.choice([0, 255])
        return rgb, t, d, mask


class LowLight(object):
    """随机降低亮度模拟低光环境"""
    def __init__(self, intensity_range=(0.3, 0.7)):
        """
        intensity_range: 亮度缩放范围，0-1
        """
        self.intensity_range = intensity_range

    def __call__(self, rgb, t, d, mask):
        factor = np.random.uniform(*self.intensity_range)
        rgb = (rgb * factor).astype(np.uint8)
        return rgb, t, d, mask
