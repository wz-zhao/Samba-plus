import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import train_macl_ddp as training
from dataset.vdt_dataset import Data as VDTDataset
from get_dataloaders_ddp import setup_all_dataloaders_for_joint_training
from tests.test_macl import config_for, data_root
from utils.data_identity import build_training_data_identity, validate_training_data_identity


def identity_for(config):
    loaders, _ = setup_all_dataloaders_for_joint_training(config)
    return build_training_data_identity(loaders, config)


def test_default_batch_size_is_two():
    assert training.parse_args([]).batch_size == 2


@pytest.mark.parametrize('mode', ['train', 'test'])
def test_vdt_visible_rgb_before_augmentation(data_root, mode):
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[..., 0] = 255
    root = data_root / 'VDT-2048' / mode.title()
    Image.fromarray(image).save(root / 'V' / '0.png')
    dataset = VDTDataset(root, mode=mode, input_size=32)
    index = next(i for i, sample in enumerate(dataset.samples) if Path(sample[0]).name == '0.png')

    def capture(rgb, thermal, depth, mask):
        np.testing.assert_array_equal(rgb[0, 0], [255, 0, 0])
        return tuple(torch.from_numpy(value.transpose(2, 0, 1).copy())
                     for value in (rgb, thermal, depth, mask))

    dataset.transform = capture
    assert dataset[index]['image'][0, 0, 0] == 255


def test_vdt_visible_rgb_normalization(data_root):
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[..., 0] = 255
    root = data_root / 'VDT-2048' / 'Test'
    Image.fromarray(image).save(root / 'V' / '0.png')
    dataset = VDTDataset(root, mode='test', input_size=32)
    index = next(i for i, sample in enumerate(dataset.samples) if Path(sample[0]).name == '0.png')
    pixel = dataset[index]['image'][:, 0, 0]
    recovered = pixel * torch.tensor([.229, .224, .225]) + torch.tensor([.485, .456, .406])
    torch.testing.assert_close(recovered, torch.tensor([1., 0., 0.]), atol=1e-6, rtol=0)


def test_unchanged_replay_identity(data_root):
    config = config_for(data_root)
    saved = identity_for(config)
    validate_training_data_identity(saved, identity_for(config))


def test_manifest_reordering_is_rejected(data_root):
    manifest = data_root / 'rgb_dataset' / 'DUTS-TR' / 'train.txt'
    rows = manifest.read_text().splitlines(keepends=True)
    rows[0] = rows[0].replace('/image.png', '/gt.png')
    manifest.write_text(''.join(rows))
    config = config_for(data_root)
    saved = identity_for(config)
    rows[0], rows[1] = rows[1], rows[0]
    manifest.write_text(''.join(rows))
    with pytest.raises(ValueError, match='data identity mismatch for DUTS-TR'):
        validate_training_data_identity(saved, identity_for(config))


def test_manifest_changes_outside_subset_are_rejected(data_root):
    config = config_for(data_root, '--max_train_samples', '2')
    saved = identity_for(config)
    manifest = data_root / 'rgb_dataset' / 'DUTS-TR' / 'train.txt'
    rows = manifest.read_text().splitlines(keepends=True)
    rows[-1] = rows[-1].replace('/image.png', '/gt.png')
    manifest.write_text(''.join(rows))
    with pytest.raises(ValueError, match='data identity mismatch for DUTS-TR'):
        validate_training_data_identity(saved, identity_for(config))


def test_changed_sample_limit_is_rejected(data_root):
    saved = identity_for(config_for(data_root))
    current = identity_for(config_for(data_root, '--max_train_samples', '2'))
    with pytest.raises(ValueError, match='max_train_samples changed'):
        validate_training_data_identity(saved, current)


def test_vdt_mapping_order_is_part_of_identity(data_root):
    config = config_for(data_root, '--train_tasks', 'VDT', '--disable_validation')
    loaders, _ = setup_all_dataloaders_for_joint_training(config)
    saved = build_training_data_identity(loaders, config)
    loaders['VDT'].dataset.dataset.samples.reverse()
    with pytest.raises(ValueError, match='data identity mismatch for VDT'):
        validate_training_data_identity(saved, build_training_data_identity(loaders, config))


@pytest.mark.parametrize('index', [-1, 4, True, 1.5])
def test_invalid_replay_indices_are_rejected(index):
    buffer = training.LossAwareReplayBuffer(10, ['DUTS-TR'])
    buffer.add('DUTS-TR', [0], [1.0])
    state = copy.deepcopy(buffer.state_dict())
    state['items']['DUTS-TR'] = {index: 2.0}
    with pytest.raises(ValueError, match='Replay index out of range'):
        buffer.load_state_dict(state, {'DUTS-TR': 4})
    assert buffer.items['DUTS-TR'] == {0: 1.0}


def test_valid_replay_indices_restore():
    buffer = training.LossAwareReplayBuffer(10, ['DUTS-TR'])
    buffer.add('DUTS-TR', [0, 3], [1.0, 2.0])
    restored = training.LossAwareReplayBuffer(10, ['DUTS-TR'])
    restored.load_state_dict(buffer.state_dict(), {'DUTS-TR': 4})
    assert restored.items == buffer.items


@pytest.mark.parametrize('legacy', [False, True])
def test_resume_rejects_bad_identity_before_model_allocation(data_root, monkeypatch, legacy):
    saved = identity_for(config_for(data_root))
    checkpoint = data_root / 'resume.pth'
    torch.save({} if legacy else {'data_identity': saved}, checkpoint)
    manifest = data_root / 'rgb_dataset' / 'DUTS-TR' / 'train.txt'
    manifest.write_text(manifest.read_text().replace('/image.png', '/gt.png', 1))
    import models.SalMamba_tri as model_module

    def should_not_allocate(**kwargs):
        pytest.fail('Model allocated before validating replay data identity')

    monkeypatch.setattr(model_module, 'Model', should_not_allocate)
    with pytest.raises(ValueError, match='no replay data identity' if legacy else 'data identity mismatch'):
        training.run(config_for(data_root, '--resume', str(checkpoint)))
