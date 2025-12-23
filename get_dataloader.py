from dvsod_dataset import Dataset as dvsod_Dataset
from rgb_dataset import Dataset as rgb_Dataset
from rgbd_dataset import Dataset as rgbd_Dataset
from vsod_dataset import Dataset as vsod_Dataset
from rgbt_dataset import Dataset as rgbt_Dataset
import transform_dvsod, transform_single, transform_rgbd, transform_vsod, transform_rgbt
from torch.utils.data import DataLoader, Subset 
from vdt_dataset import Data as Data_VDT
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler

def setup_all_dataloaders_for_joint_training(config, rank, world_size):

    if rank == 0:
        print("--- Initializing RAW DataLoaders (No Equalization) ---")

    composed_transforms_vsod_tr = transforms.Compose([
        transform_vsod.RandomFlip(), transform_vsod.RandomRotate(), transform_vsod.colorEnhance(),
        transform_vsod.randomPeper(), 
        transform_vsod.FixedResize(size=(config.input_size, config.input_size)),
        transform_vsod.Normalize(rgb_mean=[0.485, 0.456, 0.406], rgb_std=[0.229, 0.224, 0.225],
                                 flow_mean=[72.0/255, 70.8/255, 72.0/255], flow_std=[108/255, 105.8/255, 108.3/255]), 
        transform_vsod.ToTensor()
    ])

    composed_transforms_dvsod_tr = transforms.Compose([
        transform_dvsod.RandomFlip(), transform_dvsod.RandomRotate(), transform_dvsod.colorEnhance(),
        transform_dvsod.randomPeper(), 
        transform_dvsod.FixedResize(size=(config.input_size, config.input_size)),
        transform_dvsod.Normalize(rgb_mean=[0.485, 0.456, 0.406], rgb_std=[0.229, 0.224, 0.225],
                                  flow_mean=[72.0/255, 70.8/255, 72.0/255], flow_std=[108/255, 105.8/255, 108.3/255], 
                                  depth_mean=[55.8/255, 55.8/255, 55.8/255], depth_std=[92.6/255, 92.6/255, 92.6/255]), 
                                  transform_dvsod.ToTensor()])

    composed_transforms_rgb_tr  = transforms.Compose([
        transform_single.RandomFlip(), transform_single.RandomRotate(), transform_single.colorEnhance(),
        transform_single.randomPeper(), 
        transform_single.FixedResize(size=(config.input_size, config.input_size)),
        transform_single.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        transform_single.ToTensor()
    ])
    
    composed_transforms_rgbd_tr = transforms.Compose([
        transform_rgbd.RandomRotate(),
        transform_rgbd.RandomFlip(), transform_rgbd.colorEnhance(),
        transform_rgbd.randomPeper(), 
        transform_rgbd.FixedResize(size=(config.input_size, config.input_size)),
        transform_rgbd.Normalize(rgb_mean=[0.485, 0.456, 0.406], rgb_std=[0.229, 0.224, 0.225], depth_mean=[55.8/255, 55.8/255, 55.8/255], depth_std=[92.6/255, 92.6/255, 92.6/255]), 
        transform_rgbd.ToTensor(),
])

    composed_transforms_rgbt_tr = transforms.Compose([
        transform_rgbt.RandomFlip(), transform_rgbt.RandomRotate(), transform_rgbt.colorEnhance(),
        transform_rgbt.randomPeper(), 
        transform_rgbt.FixedResize(size=(config.input_size, config.input_size)),
        transform_rgbt.Normalize(rgb_mean=[0.485, 0.456, 0.406], rgb_std=[0.229, 0.224, 0.225], thermal_mean=[53.8/255, 31.3/255, 21.1/255],  thermal_std=[91.7/255, 62.3/255, 48.3/255]), 
        transform_rgbt.ToTensor()
    ])
    
    dataset_rdvs = dvsod_Dataset(datasets=['RDVS'], transform=composed_transforms_dvsod_tr, mode='train')
    dataset_vidsod = dvsod_Dataset(datasets=['vidsod_100'], transform=composed_transforms_dvsod_tr, mode='train')
    dataset_dvisal = dvsod_Dataset(datasets=['DVisal'], transform=composed_transforms_dvsod_tr, mode='train')
    dataset_dvsod_train = data.ConcatDataset([dataset_rdvs,dataset_vidsod,dataset_dvisal])

    img_root_train = '/home/user0/BRL/fabian/BRL/zhaowenzhuo/Sigma/VDT-2048 dataset/Train/'
    img_root_test = '/home/user0/BRL/fabian/BRL/zhaowenzhuo/Sigma/VDT-2048 dataset/Test/'

    train_datasets_raw = {
        'DUTS-TR': rgb_Dataset(datasets=['DUTS-TR'], transform=composed_transforms_rgb_tr, mode='train'),
        'VSOD_train': vsod_Dataset(datasets=['DAVIS','DAVSOD','FBMS'], transform=composed_transforms_vsod_tr, mode='train'),
        'vidsod_100': dvsod_Dataset(datasets=['DVisal'], transform=composed_transforms_dvsod_tr, mode='train'),
        'DVisal': dvsod_Dataset(datasets=['RDVS','DVisal','vidsod_100'], transform=composed_transforms_dvsod_tr, mode='train'),
        'RDVS': dvsod_Dataset(datasets=['RDVS'], transform=composed_transforms_dvsod_tr, mode='train'),
        'train_DUT': rgbd_Dataset(datasets=['train_DUT'], transform=composed_transforms_rgbd_tr, mode='train'),
        'VT_train': rgbt_Dataset(datasets=['VT_train'], transform=composed_transforms_rgbt_tr, mode='train'),
        'VDT': Data_VDT(img_root_train, mode='train')
        }
    all_train_dataloaders_dict = {}
    if rank == 0:
        print("\n--- Creating Training DataLoaders ---")

    for name, ds in train_datasets_raw.items():
        if len(ds) == 0:
            if rank == 0: print(f"Skipping empty dataset: {name}")
            continue
        train_sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
        
        
        dataloader = DataLoader(
            ds,
            batch_size=config.batch_size,
            num_workers=config.num_thread,
            drop_last=True,
            sampler=train_sampler
        )
        all_train_dataloaders_dict[name] = dataloader
        if rank == 0:
            print(f"  DataLoader '{name}': Original Dataset Size={len(ds)}, DataLoader Size={len(dataloader)}")


    if rank == 0:
        print("\n--- Initializing Testing Datasets ---")


    composed_transforms_vsod_te = transforms.Compose([
        transform_vsod.FixedResize(size=(config.input_size, config.input_size)),
        transform_vsod.Normalize(rgb_mean=[0.485, 0.456, 0.406], rgb_std=[0.229, 0.224, 0.225],
                                 flow_mean=[72.0/255, 70.8/255, 72.0/255], flow_std=[108/255, 105.8/255, 108.3/255]), 
        transform_vsod.ToTensor()])
    
    composed_transforms_dvsod_te = transforms.Compose([
        transform_dvsod.FixedResize(size=(config.input_size, config.input_size)),
        transform_dvsod.Normalize(rgb_mean=[0.485, 0.456, 0.406], rgb_std=[0.229, 0.224, 0.225],
                                  flow_mean=[72.0/255, 70.8/255, 72.0/255], flow_std=[108/255, 105.8/255, 108.3/255], 
                                  depth_mean=[55.8/255, 55.8/255, 55.8/255], depth_std=[92.6/255, 92.6/255, 92.6/255]), 
        transform_dvsod.ToTensor()])
    
    composed_transforms_rgb_te = transforms.Compose([
        transform_single.FixedResize(size=(config.input_size, config.input_size)),
        transform_single.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transform_single.ToTensor()])
    
    composed_transforms_rgbd_te = transforms.Compose([
        transform_rgbd.FixedResize(size=(config.input_size, config.input_size)),
        transform_rgbd.Normalize(rgb_mean=[0.485, 0.456, 0.406], rgb_std=[0.229, 0.224, 0.225],  depth_mean=[55.8/255, 55.8/255, 55.8/255], depth_std=[92.6/255, 92.6/255, 92.6/255]),
        transform_rgbd.ToTensor()])
    
    composed_transforms_rgbt_te = transforms.Compose([
        transform_rgbt.FixedResize(size=(config.input_size, config.input_size)),
        transform_rgbt.Normalize(rgb_mean=[0.485, 0.456, 0.406], rgb_std=[0.229, 0.224, 0.225], thermal_mean=[53.8/255, 31.3/255, 21.1/255],  thermal_std=[91.7/255, 62.3/255, 48.3/255]), 
        transform_rgbt.ToTensor()])


parser.add_argument('--test_dataset_rgb', type=list, default=['DUTS-TE'])
parser.add_argument('--test_dataset_rgb2', type=list, default=['HKU-IS'])
parser.add_argument('--test_dataset_rgb3', type=list, default=['PASCAL-S'])
parser.add_argument('--test_dataset_rgbd', type=list, default=['NJU2K'])
parser.add_argument('--test_dataset_rgbd2', type=list, default=['NLPR'])
parser.add_argument('--test_dataset_rgbd3', type=list, default=['STERE'])
parser.add_argument('--test_dataset_rgbt', type=list, default=['VT5000'])
parser.add_argument('--test_dataset_vsod1', type=list, default=['DAVIS'])
parser.add_argument('--test_dataset_vsod2', type=list, default=['SegTrack-V2'])
parser.add_argument('--test_dataset_dvsod', type=list, default=['RDVS'])
parser.add_argument('--test_dataset_dvsod2', type=list, default=['vidsod_100'])
parser.add_argument('--test_dataset_dvsod3', type=list, default=['DVisal'])
parser.add_argument('--train_datasets_rgb', type=list, default=['DUTS-TR'])
parser.add_argument('--train_datasets_rgbd', type=list, default=['train_DUT'])
parser.add_argument('--train_datasets_rgbt', type=list, default=['VT_train'])
parser.add_argument('--train_datasets_dvsod', type=list, default=['RDVS'])

    all_test_datasets_dict = {
        'DUTS-TE': rgb_Dataset(datasets=['DUTS-TE'], transform=composed_transforms_rgb_te, mode='test'),
        'DUT-OMRON': rgb_Dataset(datasets=['DUT-OMRON'], transform=composed_transforms_rgb_te, mode='test'),
        'VT5000': rgbt_Dataset(datasets=['VT5000'], transform=composed_transforms_rgbt_te, mode='test'),
        'VT821': rgbt_Dataset(datasets=['VT821'], transform=composed_transforms_rgbt_te, mode='test'),
        'NJU2K': rgbd_Dataset(datasets=['NJU2K'], transform=composed_transforms_rgbd_te, mode='test'),
        'NLPR': rgbd_Dataset(datasets=['NLPR'], transform=composed_transforms_rgbd_te, mode='test'),
        'DAVIS': vsod_Dataset(datasets=['DAVIS'], transform=composed_transforms_vsod_te, mode='test'),
        'SegTrack-V2': vsod_Dataset(datasets=['SegTrack-V2'], transform=composed_transforms_vsod_te, mode='test'),
        'RDVS': dvsod_Dataset(datasets=['RDVS'], transform=composed_transforms_dvsod_te, mode='test'),
        'VDT': Data_VDT(img_root_test, mode='test' )
}  
    if rank == 0:
        print("\n--- Testing Datasets Summary ---")
        for name, ds in all_test_datasets_dict.items():
            print(f"  {name}: Dataset Size={len(ds)}")

    return all_train_dataloaders_dict, all_test_datasets_dict