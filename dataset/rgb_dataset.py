import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np


def randomCrop(image, label):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    label = Image.fromarray(label)
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), np.array(label.crop(random_region))


class Dataset(data.Dataset):
    def __init__(self, datasets, mode='train', transform=None, return_size=True, root='.'):
        self.root = os.fspath(root)
        self.return_size = return_size
        if type(datasets) != list:
            datasets = [datasets]
        self.datas_id = []
        self.mode = mode
        for (i, dataset) in enumerate(datasets):

            if mode == 'train':
                data_dir = os.path.join(self.root, 'rgb_dataset', dataset)
                imgset_path = data_dir + '/train.txt'

            else:
                data_dir = os.path.join(self.root, 'rgb_dataset', dataset)
                imgset_path = data_dir + '/test.txt'

            imgset_file = open(imgset_path, encoding='utf-8')

            for line in imgset_file:
                data = {}
                img_path = line.strip("\n").split()[0]
                gt_path = line.strip("\n").split()[1]
                data['img_path'] = os.path.join(data_dir, img_path.lstrip('/\\'))
                data['gt_path'] = os.path.join(data_dir, gt_path.lstrip('/\\'))
                data['split'] = 'Mamba'
                data['dataset'] = dataset
                self.datas_id.append(data)
        self.transform = transform
    #     from torchvision import transforms
    #     self.norm = transforms.Compose([
    #     transform_single.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), transform_single.ToTensor()
    # ])

    def __getitem__(self, item):

        assert os.path.exists(self.datas_id[item]['img_path']), (
            '{} does not exist'.format(self.datas_id[item]['img_path']))
        assert os.path.exists(self.datas_id[item]['gt_path']), (
            '{} does not exist'.format(self.datas_id[item]['gt_path']))

        image = Image.open(self.datas_id[item]['img_path']).convert('RGB')
        label = Image.open(self.datas_id[item]['gt_path']).convert('L')
        label = np.array(label)

        if label.max() > 0:
            label = label / 255

        w, h = image.size
        size = (h, w)

        sample = {'image': image, 'label': label}
        if self.mode == 'train':
            sample['image'], sample['label'] = randomCrop(sample['image'],sample['label'])
        else:
            pass

        if self.transform:
            sample = self.transform(sample)
        if self.return_size:
            sample['size'] = torch.tensor(size)
        name = self.datas_id[item]['gt_path'].split('/')[-1]
        sample['dataset'] = self.datas_id[item]['dataset']
        sample['split'] = self.datas_id[item]['split']
        sample['name'] = name

        return sample

    def __len__(self):
        return len(self.datas_id)
