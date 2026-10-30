"""Dataset construction shared by the single-process and DDP MACL runners."""
from pathlib import Path

from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from transform import transform_dvsod, transform_single, transform_rgbd, transform_vsod, transform_rgbt
from dataset.dvsod_dataset import Dataset as DVSODDataset
from dataset.rgb_dataset import Dataset as RGBDataset
from dataset.rgbd_dataset import Dataset as RGBDDataset
from dataset.rgbt_dataset import Dataset as RGBTDataset
from dataset.vdt_dataset import Data as VDTDataset
from dataset.vsod_dataset import Dataset as VSODDataset

TRAIN_TASKS = ('DUTS-TR', 'train_DUT', 'VT_train', 'VSOD_train', 'DVSOD_train', 'VDT')
TEST_TASKS = ('DUTS-TE', 'DUT-OMRON', 'NJU2K', 'NLPR', 'VT5000', 'VT821',
              'DAVIS', 'SegTrack-V2', 'RDVS', 'VDT')
STAGE_TASKS = {1: ('DUTS-TR',), 2: ('train_DUT', 'VT_train', 'VSOD_train'),
               3: ('DVSOD_train', 'RDVS', 'DVisal', 'vidsod_100', 'VDT')}


class IndexedDataset(Dataset):
    def __init__(self, dataset, source):
        self.dataset = dataset
        self.source = source

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        sample['sample_index'] = index
        return sample


def build_transforms(config):
    size = (config.input_size, config.input_size)
    rgb_tr = transforms.Compose([
        transform_single.RandomFlip(), transform_single.RandomRotate(), transform_single.colorEnhance(),
        transform_single.randomPeper(), transform_single.FixedResize(size=size),
        transform_single.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transform_single.ToTensor(),
    ])
    rgb_te = transforms.Compose([
        transform_single.FixedResize(size=size),
        transform_single.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transform_single.ToTensor(),
    ])
    rgbd_tr = transforms.Compose([
        transform_rgbd.RandomRotate(), transform_rgbd.RandomFlip(), transform_rgbd.colorEnhance(),
        transform_rgbd.randomPeper(), transform_rgbd.FixedResize(size=size),
        transform_rgbd.Normalize(rgb_mean=[0.485, 0.456, 0.406], rgb_std=[0.229, 0.224, 0.225],
                                 depth_mean=[55.8 / 255] * 3, depth_std=[92.6 / 255] * 3),
        transform_rgbd.ToTensor(),
    ])
    rgbd_te = transforms.Compose([
        transform_rgbd.FixedResize(size=size),
        transform_rgbd.Normalize(rgb_mean=[0.485, 0.456, 0.406], rgb_std=[0.229, 0.224, 0.225],
                                 depth_mean=[55.8 / 255] * 3, depth_std=[92.6 / 255] * 3),
        transform_rgbd.ToTensor(),
    ])
    rgbt_tr = transforms.Compose([
        transform_rgbt.RandomFlip(), transform_rgbt.RandomRotate(), transform_rgbt.colorEnhance(),
        transform_rgbt.randomPeper(), transform_rgbt.FixedResize(size=size),
        transform_rgbt.Normalize(rgb_mean=[0.485, 0.456, 0.406], rgb_std=[0.229, 0.224, 0.225],
                                 thermal_mean=[53.8 / 255, 31.3 / 255, 21.1 / 255],
                                 thermal_std=[91.7 / 255, 62.3 / 255, 48.3 / 255]),
        transform_rgbt.ToTensor(),
    ])
    rgbt_te = transforms.Compose([
        transform_rgbt.FixedResize(size=size),
        transform_rgbt.Normalize(rgb_mean=[0.485, 0.456, 0.406], rgb_std=[0.229, 0.224, 0.225],
                                 thermal_mean=[53.8 / 255, 31.3 / 255, 21.1 / 255],
                                 thermal_std=[91.7 / 255, 62.3 / 255, 48.3 / 255]),
        transform_rgbt.ToTensor(),
    ])
    vsod_tr = transforms.Compose([
        transform_vsod.RandomFlip(), transform_vsod.RandomRotate(), transform_vsod.colorEnhance(),
        transform_vsod.randomPeper(), transform_vsod.FixedResize(size=size),
        transform_vsod.Normalize(rgb_mean=[0.485, 0.456, 0.406], rgb_std=[0.229, 0.224, 0.225],
                                 flow_mean=[72.0 / 255, 70.8 / 255, 72.0 / 255],
                                 flow_std=[108 / 255, 105.8 / 255, 108.3 / 255]),
        transform_vsod.ToTensor(),
    ])
    dvsod_tr = transforms.Compose([
        transform_dvsod.RandomFlip(), transform_dvsod.RandomRotate(), transform_dvsod.colorEnhance(),
        transform_dvsod.randomPeper(), transform_dvsod.FixedResize(size=size),
        transform_dvsod.Normalize(rgb_mean=[0.485, 0.456, 0.406], rgb_std=[0.229, 0.224, 0.225],
                                  flow_mean=[72.0 / 255, 70.8 / 255, 72.0 / 255],
                                  flow_std=[108 / 255, 105.8 / 255, 108.3 / 255],
                                  depth_mean=[55.8 / 255] * 3, depth_std=[92.6 / 255] * 3),
        transform_dvsod.ToTensor(),
    ])
    return rgb_tr, rgb_te, rgbd_tr, rgbd_te, rgbt_tr, rgbt_te, vsod_tr, dvsod_tr


def setup_all_dataloaders_for_joint_training(config, rank=0, world_size=1):
    root = Path(getattr(config, 'data_root', '.')).expanduser().resolve()
    train_names = list(getattr(config, 'train_tasks', TRAIN_TASKS))
    if getattr(config, 'mode', 'train') == 'test':
        train_names = []
    test_names = [] if getattr(config, 'disable_validation', False) else list(
        getattr(config, 'test_tasks', TEST_TASKS))
    if getattr(config, 'test_dataset_rgbd', None):
        test_names = [n for n in test_names if n not in ('NJU2K', 'NLPR')]
        test_names.extend(config.test_dataset_rgbd)
    if len(set(train_names)) != len(train_names) or len(set(test_names)) != len(test_names):
        raise ValueError('Dataset task names must be unique')
    rgb_tr, rgb_te, rd_tr, rd_te, rt_tr, rt_te, vs_tr, dvs_tr = build_transforms(config)
    vs_te = transforms.Compose([transform_vsod.FixedResize((config.input_size,) * 2),
                                vs_tr.transforms[-2], transform_vsod.ToTensor()])
    dvs_te = transforms.Compose([transform_dvsod.FixedResize((config.input_size,) * 2),
                                 dvs_tr.transforms[-2], transform_dvsod.ToTensor()])
    specs = {
        'DUTS-TR': (RGBDataset, ['DUTS-TR'], 'rgb_dataset', rgb_tr),
        'train_DUT': (RGBDDataset, ['train_DUT'], 'rgbd_dataset', rd_tr),
        'VT_train': (RGBTDataset, ['VT_train'], 'rgbt_dataset', rt_tr),
        'VSOD_train': (VSODDataset, ['DAVIS', 'DAVSOD', 'FBMS'], 'vsod_dataset', vs_tr),
        'DVSOD_train': (DVSODDataset, ['RDVS', 'DVisal', 'vidsod_100'], 'dvsod_dataset', dvs_tr),
    }
    tests = {}
    for names, cls, folder, transform in [
        (['DUTS-TE', 'DUT-OMRON', 'ECSSD', 'HKU-IS', 'PASCAL-S'], RGBDataset, 'rgb_dataset', rgb_te),
        (['NJU2K', 'NLPR', 'DUT-RGBD', 'SIP', 'STERE'], RGBDDataset, 'rgbd_dataset', rd_te),
        (['VT5000', 'VT821', 'VT1000'], RGBTDataset, 'rgbt_dataset', rt_te),
        (['DAVIS', 'DAVSOD', 'FBMS', 'SegTrack-V2', 'VOS'], VSODDataset, 'vsod_dataset', vs_te),
        (['RDVS', 'DVisal', 'vidsod_100'], DVSODDataset, 'dvsod_dataset', dvs_te),
    ]:
        tests.update({name: (cls, [name], folder, transform) for name in names})
    for name in ['RDVS', 'DVisal', 'vidsod_100']:
        specs[name] = (DVSODDataset, [name], 'dvsod_dataset', dvs_tr)

    missing = []
    def make(name, mode):
        if name == 'VDT':
            attr = 'vdt_train_root' if mode == 'train' else 'vdt_test_root'
            path = Path(getattr(config, attr, '') or root / 'VDT-2048' / mode.title())
            for part in ['V', 'D', 'T', 'GT']:
                if not (path / part).is_dir():
                    missing.append(str(path / part))
            return None if missing else VDTDataset(path, mode=mode, input_size=config.input_size)
        catalog = specs if mode == 'train' else tests
        if name not in catalog:
            raise ValueError(f'Unknown {mode} dataset task: {name}')
        cls, names, folder, transform = catalog[name]
        expected = [root / folder / mode / n / f'{mode}.txt' if folder == 'vsod_dataset'
                    else root / folder / n / f'{mode}.txt' for n in names]
        absent = [str(path) for path in expected if not path.is_file()]
        missing.extend(absent)
        if absent:
            return None
        dataset = cls(names, mode=mode, transform=transform, root=root)
        if len(dataset) == 0:
            raise ValueError(f'{mode} dataset {name} is empty')
        return dataset

    raw = {name: make(name, 'train') for name in train_names}
    evaluation = {name: make(name, 'test') for name in test_names}
    if missing:
        raise FileNotFoundError('Missing selected dataset resources under --data_root:\n  '
                                + '\n  '.join(missing)
                                + '\nSelect available --train_tasks/--test_tasks or set the correct root.')
    train_limit = getattr(config, 'max_train_samples', 0)
    test_limit = getattr(config, 'max_test_samples', 0)
    if train_limit:
        raw = {name: Subset(ds, range(min(len(ds), train_limit))) for name, ds in raw.items()}
    if test_limit:
        evaluation = {name: Subset(ds, range(min(len(ds), test_limit)))
                      for name, ds in evaluation.items()}
    loaders = {}
    for name, dataset in raw.items():
        dataset = IndexedDataset(dataset, name)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        loader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler,
                            num_workers=config.num_thread, drop_last=True, pin_memory=True)
        if len(loader) == 0:
            raise ValueError(f'{name} has no full batch per rank; lower --batch_size or world size')
        loaders[name] = loader
        if rank == 0:
            print(f'{name}: {len(dataset)} samples, {len(loader)} batches per rank')
    return loaders, evaluation
