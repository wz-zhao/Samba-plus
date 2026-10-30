import json
import random
import shutil
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

import train_macl_ddp as training
from get_dataloaders_ddp import setup_all_dataloaders_for_joint_training
from loss import total_loss_suite_optimized, structure_loss_edge
from Smeasure import s_measure
from utils.init_func import group_weight
from val_join_ddp import evaluate_all_tests_and_save_best


class TinyModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.rgb = nn.Conv2d(3, 1, 1)
        self.flow = nn.Conv2d(3, 1, 1)
        self.depth = nn.Conv2d(3, 1, 1)
        self.thermal = nn.Conv2d(3, 1, 1)

    def forward(self, rgb, flow=None, depth=None, thermal=None, **kwargs):
        out = self.rgb(rgb)
        for value, layer in [(flow, self.flow), (depth, self.depth), (thermal, self.thermal)]:
            if value is not None:
                out = out + layer(value)
        return out, out, {}


def make_data(root, count=20):
    rng = np.random.default_rng(9)
    image = rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:30, 10:30] = 255
    catalogs = [('rgb_dataset', 'DUTS-TR', 'train', 2),
                ('rgb_dataset', 'DUTS-TE', 'test', 2),
                ('rgbd_dataset', 'train_DUT', 'train', 3),
                ('rgbd_dataset', 'NJU2K', 'test', 3),
                ('rgbd_dataset', 'NLPR', 'test', 3),
                ('rgbt_dataset', 'VT_train', 'train', 3),
                ('rgbt_dataset', 'VT5000', 'test', 3)]
    catalogs += [('vsod_dataset', name, 'train', 3) for name in ['DAVIS', 'DAVSOD', 'FBMS']]
    catalogs += [('dvsod_dataset', name, 'train', 4) for name in ['RDVS', 'DVisal', 'vidsod_100']]
    catalogs += [('vsod_dataset', 'DAVIS', 'test', 3), ('dvsod_dataset', 'RDVS', 'test', 4)]
    for folder, name, split, columns in catalogs:
        directory = root / folder / split / name if folder == 'vsod_dataset' else root / folder / name
        # The DVisal adapter maps its legacy paths into vidsod_100.
        assets = directory / 'seq' / 'Images'
        assets.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(assets / 'image.png')
        Image.fromarray(mask).save(assets / 'gt.png')
        row = ['/seq/Images/image.png', '/seq/Images/gt.png'] + ['/seq/Images/image.png'] * (columns - 2)
        (directory / f'{split}.txt').write_text((' '.join(row) + '\n') * count, encoding='utf-8')
    for split in ['Train', 'Test']:
        for folder in ['V', 'D', 'T', 'GT']:
            directory = root / 'VDT-2048' / split / folder
            directory.mkdir(parents=True, exist_ok=True)
            for index in range(4):
                Image.fromarray(np.repeat(mask[..., None], 3, axis=2) if folder == 'GT' else image).save(directory / f'{index}.png')


@pytest.fixture
def data_root(tmp_path):
    make_data(tmp_path)
    return tmp_path


def config_for(root, *extra):
    validation_options = ['--stage_patience', '0'] if '--disable_validation' in extra else []
    return training.parse_args(['--data_root', str(root), '--device', 'cpu', '--from_scratch',
        '--input_size', '32', '--batch_size', '2', '--num_thread', '0', '--epoch', '3',
        '--stage1_epochs', '1', '--stage2_epochs', '1', '--save_fold', str(root / 'checkpoints'),
        '--train_tasks', 'DUTS-TR', 'train_DUT', 'DVSOD_train', '--test_tasks', 'DUTS-TE',
        '--accumulation_steps', '4', '--validation_frequency', '0', *validation_options, *extra])


@pytest.mark.parametrize('batch_size', [1, 3, 7])
def test_sample_replay_quotas(batch_size):
    for rgb, dual in [(0, 0), (.1, 0), (.1, .3)]:
        schedule = training.ReplaySchedule(rgb, dual)
        counts = Counter()
        for _ in range(100):
            current = schedule.next(batch_size)
            assert sum(current.values()) == batch_size
            counts.update(current)
        assert counts['rgb'] == 10 * batch_size if rgb else counts['rgb'] == 0
        assert counts['dual'] == 30 * batch_size if dual else counts['dual'] == 0


def test_loss_is_per_sample():
    pred = torch.randn(3, 1, 32, 32, requires_grad=True)
    gt = torch.rand_like(pred).round()
    for function in [total_loss_suite_optimized, structure_loss_edge]:
        per_sample = function(pred, gt, reduction='none')
        assert per_sample.shape == (3,)
        torch.testing.assert_close(per_sample.mean(), function(pred, gt))
        torch.testing.assert_close(per_sample[1], function(pred[1:2], gt[1:2]))


def test_optimizer_covers_direct_and_shared_parameters():
    model = TinyModel()
    model.A_logs = nn.Parameter(torch.randn(2, 3))
    model.A_logs._no_weight_decay = True
    model.custom_weight = nn.Parameter(torch.randn(3, 3))
    groups = group_weight([], model, nn.BatchNorm2d, 1e-4)
    ids = [id(param) for group in groups for param in group['params']]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    assert id(model.A_logs) in {id(p) for p in groups[1]['params']}
    assert id(model.custom_weight) in {id(p) for p in groups[0]['params']}


def test_only_selected_datasets_are_required(data_root):
    config = config_for(data_root, '--train_tasks', 'DUTS-TR', '--disable_validation')
    shutil.rmtree(data_root / 'dvsod_dataset')
    loaders, tests = setup_all_dataloaders_for_joint_training(config)
    assert list(loaders) == ['DUTS-TR'] and not tests
    batch = next(iter(loaders['DUTS-TR']))
    assert batch['image'].shape == (2, 3, 32, 32)
    assert batch['label'].shape == (2, 1, 32, 32)


def test_real_data_subset_limits(data_root):
    config = config_for(data_root, '--max_train_samples', '6', '--max_test_samples', '3')
    loaders, tests = setup_all_dataloaders_for_joint_training(config)
    assert all(len(loader.dataset) == 6 for loader in loaders.values())
    assert len(tests['DUTS-TE']) == 3
    batch = next(iter(loaders['DUTS-TR']))
    assert batch['sample_index'].max() < 6


def test_all_modalities_load(data_root):
    config = config_for(data_root, '--train_tasks', 'DUTS-TR', 'train_DUT', 'VT_train', 'VSOD_train', 'DVSOD_train', 'VDT',
                        '--test_tasks', 'DUTS-TE', 'NJU2K', 'NLPR', 'VT5000', 'DAVIS', 'RDVS', 'VDT')
    loaders, tests = setup_all_dataloaders_for_joint_training(config)
    for name, loader in loaders.items():
        batch = next(iter(loader))
        assert batch['image'].shape == (2, 3, 32, 32), name
        assert batch['label'].shape == (2, 1, 32, 32), name
        assert torch.isfinite(batch['image']).all()
        for key in ['image', 'label', 'depth', 'flow', 'thermal']:
            if key in batch:
                assert batch[key].dtype == torch.float32, (name, key)
    for name, dataset in tests.items():
        assert dataset[0]['image'].shape == (3, 32, 32), name


def test_missing_data_lists_actionable_paths(tmp_path):
    with pytest.raises(FileNotFoundError, match='rgb_dataset.*DUTS-TR'):
        setup_all_dataloaders_for_joint_training(config_for(tmp_path))


def test_replay_prefers_hard_examples_and_keeps_modalities(data_root):
    loaders, _ = setup_all_dataloaders_for_joint_training(config_for(data_root))
    datasets = {name: loader.dataset for name, loader in loaders.items()}
    buffer = training.LossAwareReplayBuffer(10, ['train_DUT'])
    buffer.add('train_DUT', [0, 1], [0.01, 10])
    batch, indices = buffer.sample('train_DUT', 100, datasets, random.Random(0))
    assert indices.count(1) > 95
    assert 'depth' in batch and 'flow' not in batch and 'thermal' not in batch
    assert len(buffer) == 2


def test_smeasure_handles_small_regions():
    for shape in [(1, 1), (1, 4), (4, 4)]:
        for target in [torch.zeros(shape), torch.ones(shape)]:
            torch.testing.assert_close(s_measure(target, target), torch.tensor(1.0))
        target = torch.zeros(shape)
        target.reshape(-1)[0] = 1
        assert torch.isfinite(s_measure(torch.rand(shape), target))


def test_single_process_evaluation(data_root):
    config = config_for(data_root)
    _, tests = setup_all_dataloaders_for_joint_training(config)
    model = TinyModel().train()
    score = evaluate_all_tests_and_save_best(model, tests, 0, data_root / 'eval', config)
    assert 0 <= score <= 1 and model.training
    assert (data_root / 'eval' / 'validation.jsonl').is_file()


def test_real_rq_and_graph_and_deform_conv(monkeypatch):
    from models.SalMamba_tri import RandomModule
    from ERM import GNNRoIFusion, deform_conv2d_pytorch_grouped
    x = torch.linspace(-2, 2, 64).reshape(1, 1, 8, 8).repeat(2, 3, 1, 1)
    mask = torch.ones(2, 1, 8, 8)
    module = RandomModule(modality='Depth', spacing='log', salient_max_delta=.01)
    output = module(x, mask)
    assert torch.isfinite(output).all() and (output - x).abs().max() <= .040001
    graph = GNNRoIFusion(8)
    edges = graph._get_or_create_edge_index(3, 4, torch.device('cpu'))
    assert torch.equal(edges[0] // 3, edges[1] // 3)
    for size in [1, 4]:
        image = torch.randn(2, 4, size, size)
        weight = torch.randn(4, 2, 3, 3)
        offset = torch.zeros(2, 36, size, size)
        mask = torch.ones(2, 18, size, size)
        actual = deform_conv2d_pytorch_grouped(image, offset, weight, padding=(1, 1), mask=mask, groups=2)
        expected = F.conv2d(image, weight, padding=1, groups=2)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_three_stage_training_and_resume(data_root, monkeypatch):
    import models.SalMamba_tri as model_module
    monkeypatch.setattr(model_module, 'Model', TinyModel)
    original_save = training.save_checkpoint
    snapshot = data_root / 'stage1_resume.pth'
    def save(*args, **kwargs):
        original_save(*args, **kwargs)
        if args[5] == 0 and kwargs.get('filename', 'last_checkpoint.pth') == 'last_checkpoint.pth':
            shutil.copyfile(Path(args[8].save_fold) / 'last_checkpoint.pth', snapshot)
    monkeypatch.setattr(training, 'save_checkpoint', save)
    config = config_for(data_root)
    training.run(config)
    directory = Path(config.save_fold)
    rows = [json.loads(line) for line in (directory / 'training_log_macl.jsonl').read_text().splitlines()]
    assert [row['stage'] for row in rows] == [1, 2, 3]
    assert rows[0]['samples'] == {'current': 20}
    assert rows[1]['samples']['rgb'] == 2
    assert rows[2]['samples']['rgb'] == 6 and rows[2]['samples']['dual'] == 18
    config = config_for(data_root, '--resume', str(snapshot), '--save_fold', str(data_root / 'resumed'))
    training.run(config)
    resumed = [json.loads(line) for line in (Path(config.save_fold) / 'training_log_macl.jsonl').read_text().splitlines()]
    assert [row['stage'] for row in resumed] == [2, 3]
    assert (Path(config.save_fold) / 'last_checkpoint.pth').exists()
    original = torch.load(directory / 'last_checkpoint.pth', map_location='cpu', weights_only=False)
    continued = torch.load(Path(config.save_fold) / 'last_checkpoint.pth', map_location='cpu', weights_only=False)
    assert original['data_identity'] == continued['data_identity']
    assert original['data_identity']['version'] == 1
    for name in original['model']:
        torch.testing.assert_close(original['model'][name], continued['model'][name], atol=0, rtol=0)


def test_training_all_task_schemas(tmp_path, monkeypatch):
    import models.SalMamba_tri as model_module
    monkeypatch.setattr(model_module, 'Model', TinyModel)
    make_data(tmp_path, count=4)
    config = config_for(tmp_path, '--train_tasks', 'DUTS-TR', 'train_DUT', 'VT_train',
                        'VSOD_train', 'DVSOD_train', 'VDT', '--disable_validation')
    training.run(config)
    rows = [json.loads(line) for line in (Path(config.save_fold) / 'training_log_macl.jsonl').read_text().splitlines()]
    assert [row['stage'] for row in rows] == [1, 2, 3]
    for row in rows[1:]:
        total = sum(row['samples'].values())
        assert abs(row['samples'].get('rgb', 0) - total * .1) <= 1
        if row['stage'] == 3:
            assert abs(row['samples']['dual'] - total * .3) <= 1
