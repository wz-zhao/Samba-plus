#coding=utf-8

import os
import cv2
import numpy as np
import torch
from . import transform

from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")
#BGR
# mean_rgb = np.array([[[0.391, 0.363, 0.338]]])
# mean_t =np.array([[[0.170,  0.403, 0.556]]])
# mean_d =np.array([[[0.034,  0.034, 0.034]]])
# std_rgb = np.array([[[0.224 , 0.217 , 0.206 ]]])
# std_t = np.array([[[0.160 , 0.188 , 0.238 ]]])
# std_d = np.array([[[0.007 , 0.007 , 0.007 ]]])


mean_rgb = np.array([[[0.485, 0.456, 0.406]]])
std_rgb = np.array([[[0.229, 0.224, 0.225]]])
mean_t =np.array([[[0.170,  0.403, 0.556]]])
mean_d =np.array([[[0.034,  0.034, 0.034]]])
std_t = np.array([[[0.160 , 0.188 , 0.238 ]]])
std_d = np.array([[[0.007 , 0.007 , 0.007 ]]])




class Data(Dataset):
    def __init__(self, root, mode='train'):
        self.samples = []
        lines = os.listdir(os.path.join(root, 'GT'))
        self.mode = mode
        for line in lines:
            rgbpath = os.path.join(root, 'V', line[:-4]+'.png')
            tpath = os.path.join(root, 'T', line[:-4]+'.png')
            dpath = os.path.join(root, 'D', line[:-4] + '.png')
            maskpath = os.path.join(root, 'GT', line)
            self.samples.append([rgbpath,tpath,dpath,maskpath])

        if mode == 'train':
            self.transform = transform.Compose( 
                                                transform.ColorEnhance(),
                                                transform.Resize(448, 448), transform.Random_rotate(),
                                                transform.RandomHorizontalFlip(), 
                                                transform.Normalize(mean1=mean_rgb, mean2=mean_t, mean3=mean_d, std1=std_rgb, std2=std_t, std3=std_d),
                                                transform.ToTensor())
        elif mode == 'test':
            self.transform = transform.Compose(transform.Normalize(mean1=mean_rgb, mean2=mean_t, mean3=mean_d, std1=std_rgb, std2=std_t, std3=std_d),
                                                transform.Resize(448, 448),
                                                transform.ToTensor())
        else:
            raise ValueError

    def __getitem__(self, idx):
        rgbpath,tpath,dpath,maskpath = self.samples[idx]
        rgb = cv2.imread(rgbpath).astype(np.float32)
        
        t = cv2.imread(tpath).astype(np.float32)
        d = cv2.imread(dpath).astype(np.float32)
        mask = cv2.imread(maskpath).astype(np.float32)
       
        H, W, C = mask.shape
        rgb,t,d,mask = self.transform(rgb,t,d,mask)

        if mask.max() > 1:
            mask = mask / 255
        # if  self.mode == 'train':
        #     rgb,t,d = getRandomSample(rgb,t,d)
            
        rgb = rgb.float() 
        
        t = t.float() 
        d = d.float()
        mask = mask.float()
        
        sample = {'image': rgb, 'label': mask, 'depth': d, 'thermal': t}
        return sample
        # return sample,(H, W), maskpath.split('/')[-1]

    def __len__(self):
        return len(self.samples)

