import torch
import math
import numbers
import random
import numpy as np
import cv2

from PIL import Image, ImageOps, ImageEnhance


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size  # h, w
        self.padding = padding

    def __call__(self, sample):
        img, mask, flow = sample['image'], sample['label'], sample['flow']

        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)
            flow = ImageOps.expand(flow, border=self.padding, fill=0)

        assert img.size == mask.size
        assert img.size == flow.size
        w, h = img.size
        th, tw = self.size  # target size
        if w == tw and h == th:
            return {'image': img,
                    'label': mask,
                    'flow': flow}
        if w < tw or h < th:
            img = img.resize((tw, th), Image.BILINEAR)
            mask = mask.resize((tw, th), Image.NEAREST)
            flow = flow.resize((tw, th), Image.BILINEAR)
            return {'image': img,
                    'label': mask,
                    'flow': flow}

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        mask = mask.crop((x1, y1, x1 + tw, y1 + th))
        flow = flow.crop((x1, y1, x1 + tw, y1 + th))

        return {'image': img,
                'label': mask,
                'flow': flow}


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        flow = sample['flow']
        assert img.size == mask.size
        assert img.size == flow.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        mask = mask.crop((x1, y1, x1 + tw, y1 + th))
        flow = flow.crop((x1, y1, x1 + tw, y1 + th))

        return {'image': img,
                'label': mask,
                'flow': flow}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        depth = sample['depth']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = Image.fromarray(mask)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            mask = np.array(mask)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask,
                'depth': depth}


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, rgb_mean, rgb_std, depth_mean, depth_std):
        self.rgb_mean = np.array(rgb_mean, dtype=np.float32)
        self.rgb_std = np.array(rgb_std, dtype=np.float32)
        self.depth_mean = np.array(depth_mean, dtype=np.float32)
        self.depth_std = np.array(depth_std, dtype=np.float32)

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        mask = np.array(sample['label']).astype(np.float32)
        depth = np.array(sample['depth']).astype(np.float32)
        img /= 255.0
        img -= self.rgb_mean
        img /= self.rgb_std

        depth /= 255.0
        depth -= self.depth_mean
        depth /= self.depth_std

        return {'image': img,
                'label': mask,
                'depth': depth}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = np.array(sample['image']).astype(np.float32).transpose((2, 0, 1))
        depth = np.array(sample['depth']).astype(np.float32).transpose((2, 0, 1))
        mask = np.expand_dims(sample['label'].astype(np.float32), -1).transpose((2, 0, 1))
        mask[mask == 255] = 0

        img = torch.from_numpy(img).float()
        depth = torch.from_numpy(depth).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask,
                'depth': depth}


class FixedResize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        depth = sample['depth']

        img = img.resize(self.size, Image.BILINEAR)
        mask = cv2.resize(mask, self.size, cv2.INTER_NEAREST)
        depth = depth.resize(self.size, Image.BILINEAR)

        return {'image': img,
                'label': mask,
                'depth': depth}


class Scale(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size
        assert img.size == flow.size

        w, h = img.size

        if (w >= h and w == self.size[1]) or (h >= w and h == self.size[0]):
            return {'image': img,
                    'label': mask,
                    'flow': flow}
        oh, ow = self.size
        img = img.resize((ow, oh), Image.BILINEAR)
        flow = flow.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        return {'image': img,
                'label': mask,
                'flow': flow}


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        flow = sample['flow']
        edge = sample['edge']
        assert img.size == mask.size
        assert img.size == flow.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                flow = flow.crop((x1, y1, x1 + w, y1 + h))

                assert (img.size == (w, h))

                img = img.resize((self.size, self.size), Image.BILINEAR)
                mask = mask.resize((self.size, self.size), Image.NEAREST)
                flow = flow.resize((self.size, self.size), Image.BILINEAR)

                return {'image': img,
                        'label': mask,
                        'flow': flow}

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        sample = crop(scale(sample))
        return sample

class RandomRotateOrthogonal(object):

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        flow = sample['flow']

        rotate_degree = random.randint(0, 3) * 90
        if rotate_degree > 0:
            img = img.rotate(rotate_degree, Image.BILINEAR)
            mask = mask.rotate(rotate_degree, Image.NEAREST)
            flow = flow.rotate(rotate_degree, Image.BILINEAR)

        return {'image': img,
                'label': mask,
                'flow': flow}


class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        flow = sample['flow']
        assert img.size == mask.size
        assert img.size == flow.size

        w = int(random.uniform(0.8, 2.5) * img.size[0])
        h = int(random.uniform(0.8, 2.5) * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)
        flow = flow.resize((w, h), Image.BILINEAR)

        sample = {'image': img, 'label': mask, 'flow': flow}

        return self.crop(self.scale(sample))


class RandomScale(object):
    def __init__(self, limit):
        self.limit = limit

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        flow = sample['flow']
        assert img.size == mask.size
        assert img.size == flow.size

        scale = random.uniform(self.limit[0], self.limit[1])
        w = int(scale * img.size[0])
        h = int(scale * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)
        flow = flow.resize((w, h), Image.BILINEAR)

        return {'image': img, 'label': mask, 'flow': flow}


class RandomRotate(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        depth = sample['depth']
        mode = Image.BICUBIC
        label = Image.fromarray(label)
        if random.random() > 0.8:
            random_angle = np.random.randint(-15, 15)
            image = image.rotate(random_angle, mode)
            label = label.rotate(random_angle, mode)
            depth = depth.rotate(random_angle, mode)
        label = np.array(label)
        return {'image': image,
                'label': label,
                'depth': depth}


class colorEnhance(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        depth = sample['depth']
        bright_intensity = random.randint(5, 15) / 10.0
        image = ImageEnhance.Brightness(image).enhance(bright_intensity)
        contrast_intensity = random.randint(5, 15) / 10.0
        image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
        color_intensity = random.randint(0, 20) / 10.0
        image = ImageEnhance.Color(image).enhance(color_intensity)
        sharp_intensity = random.randint(0, 30) / 10.0
        image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
        return {'image': image,
                'label': label,
                'depth': depth}


class randomPeper(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        depth = sample['depth']
        noiseNum = int(0.0015 * label.shape[0] * label.shape[1])
        for i in range(noiseNum):
            randX = random.randint(0, label.shape[0] - 1)
            randY = random.randint(0, label.shape[1] - 1)
            if random.randint(0, 1) == 0:
                label[randX, randY] = 0
            else:
                label[randX, randY] = 1
        return {'image': image,
                'label': label,
                'depth': depth}


class RandomFlip(object):
    def __call__(self, sample):
        img = sample['image']
        label = sample['label']
        depth = sample['depth']
        flip_flag = random.randint(0, 1)
        # flip_flag2= random.randint(0,1)
        #left right flip
        if flip_flag == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label = Image.fromarray(label)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
            label = np.array(label)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        #top bottom flip
        # if flip_flag2==1:
        #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
        #     label = label.transpose(Image.FLIP_TOP_BOTTOM)
        #     depth = depth.transpose(Image.FLIP_TOP_BOTTOM)

        return {'image': img,
                'label': label,
                'depth': depth}

    
    
# ==============================================================================
# <<< 在这里添加新的数据增强类 >>>
# ==============================================================================
from torchvision.transforms import functional as F
from torchvision import transforms

# ==============================================================================
# <<< 在这里添加新的、带概率的随机裁切类 >>>
# ==============================================================================
from torchvision.transforms import functional as F
from torchvision import transforms

class ProbabilisticRandomResizedCrop(object):
    """
    以一定的概率 p 对样本进行随机裁切并缩放。
    如果不执行，则会进行一个简单的确定性缩放，以保证输出尺寸统一。
    """
    def __init__(self, size, p=1.0, scale=(0.5, 1.0), ratio=(3. / 4., 4. / 3.)):
        """
        Args:
            size (int or tuple): 最终输出的尺寸。
            p (float): 执行随机裁切并缩放的概率，默认为1.0（总是执行）。
            scale (tuple): 随机裁切面积的范围。
            ratio (tuple): 随机裁切长宽比的范围。
        """
        self.size = (size, size) if isinstance(size, int) else size
        self.p = p
        self.scale = scale
        self.ratio = ratio

    def __call__(self, sample):
        img = sample['image']
        depth = sample.get('depth')
        label = sample['label']
        
        if not isinstance(label, Image.Image):
            label = Image.fromarray(label)

        # 以概率 p 决定是否执行随机裁切
        if random.random() < self.p:
            # --- 执行随机裁切 (RandomResizedCrop) 的逻辑 ---
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                img, scale=self.scale, ratio=self.ratio)

            img = F.resized_crop(img, i, j, h, w, self.size, Image.BILINEAR)
            label = F.resized_crop(label, i, j, h, w, self.size, Image.NEAREST)
            if depth is not None:
                depth = F.resized_crop(depth, i, j, h, w, self.size, Image.BILINEAR)
        else:
            # --- Fallback: 执行简单的确定性缩放 ---
            # 这里的逻辑类似于您之前的 FixedResize
            img = img.resize(self.size, Image.BILINEAR)
            label = label.resize(self.size, Image.NEAREST)
            if depth is not None:
                depth = depth.resize(self.size, Image.BILINEAR)

        # 更新 sample 字典
        sample['image'] = img
        sample['label'] = np.array(label)
        if depth is not None:
            sample['depth'] = depth

        return sample
    
class MyRandomErasing(object):
    """
    对样本中的 image 和 depth Tensor 进行同步的随机擦除。
    这是一个强大的正则化方法，应在 ToTensor() 之后使用。
    """
    def __init__(self, p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0):
        # 初始化一个 torchvision 的 RandomErasing 实例
        self.eraser = transforms.RandomErasing(p=p, scale=scale, ratio=ratio, value=value, inplace=False)

    def __call__(self, sample):
        # 这个变换作用于 Tensor，所以确保它在 ToTensor() 之后被调用
        img_tensor = sample['image']
        depth_tensor = sample['depth']

        # 分别对 image 和 depth Tensor 应用擦除
        # 注意：这里的擦除位置是随机且独立的，模拟了不同模态可能出现的不同遮挡
        sample['image'] = self.eraser(img_tensor)
        sample['depth'] = self.eraser(depth_tensor)
        
        # label Tensor 不需要进行擦除
        return sample
# ==============================================================================
# <<< 新增代码结束 >>>
# ==============================================================================

from torchvision.transforms import functional as F
from torchvision import transforms
import numbers
from PIL import Image
import numpy as np
import random # Make sure random is imported

class MyRandomAffine(object):
    """
    对 image, depth, label 进行同步的随机仿射变换。(已修复版本兼容性问题)
    """
    def __init__(self, degrees, translate=None, scale=None, shear=None):
        self.degrees = (-degrees, degrees) if isinstance(degrees, numbers.Number) else degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def __call__(self, sample):
        img = sample['image']
        depth = sample.get('depth')
        label = sample['label']
        
        if not isinstance(label, Image.Image):
            label = Image.fromarray(label)

        # 获取将要应用在所有图像上的、相同的随机变换参数
        angle, translations, scale_factor, shear_factor = transforms.RandomAffine.get_params(
            self.degrees, self.translate, self.scale, self.shear, img.size)

        # <<< MODIFICATION START >>>
        # Using PIL.Image integer constants for interpolation, which is compatible with more torchvision versions.
        
        # Apply transformation to image and depth using bilinear interpolation
        img = F.affine(img, angle, translations, scale_factor, shear_factor, interpolation=Image.BILINEAR, fill=0)
        if depth is not None:
            depth = F.affine(depth, angle, translations, scale_factor, shear_factor, interpolation=Image.BILINEAR, fill=0)
        
        # Apply transformation to label, which MUST use nearest neighbor interpolation
        label = F.affine(label, angle, translations, scale_factor, shear_factor, interpolation=Image.NEAREST, fill=0)
        # <<< MODIFICATION END >>>

        sample['image'] = img
        sample['depth'] = depth
        sample['label'] = np.array(label)

        return sample