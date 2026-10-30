"""MACL: RGB anchor -> one auxiliary modality -> multiple auxiliaries.

All entry points share this implementation. Replay quotas count samples rather
than replacing entire batches; buffers store dataset references and per-sample
losses, retaining hard examples without storing gigabytes of image tensors.
"""
import argparse
import json
import math
import os
import random
import shutil
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import default_collate
from tqdm import tqdm

from get_dataloaders_ddp import (STAGE_TASKS, TEST_TASKS, TRAIN_TASKS,
                                 setup_all_dataloaders_for_joint_training)
from loss import structure_loss_edge, total_loss_suite_optimized
from utils.init_func import group_weight
from utils.data_identity import build_training_data_identity, validate_training_data_identity
from val_join_ddp import evaluate_all_tests_and_save_best


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', type=int, default=500,
                        help='Sum of stage epoch caps; early switching may finish sooner')
    parser.add_argument('--stage1_epochs', type=int, default=80, help='Maximum epochs in Stage 1')
    parser.add_argument('--stage2_epochs', type=int, default=180, help='Maximum epochs in Stage 2')
    parser.add_argument('--stage_patience', type=int, default=5,
                        help='Advance after N consecutive epochs without improvement; 0 disables early switching')
    parser.add_argument('--stage_min_delta', type=float, default=0.0,
                        help='Required absolute improvement in the stage validation S-measure')
    parser.add_argument('--epoch_save', type=int, default=5)
    parser.add_argument('--save_fold', default='./checkpoints')
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--test_batch_size', type=int, default=2)
    parser.add_argument('--max_train_samples', type=int, default=0,
                        help='Use at most this many samples per training task; 0 uses all')
    parser.add_argument('--max_test_samples', type=int, default=0,
                        help='Use at most this many samples per test dataset; 0 uses all')
    parser.add_argument('--max_steps_per_epoch', type=int, default=0,
                        help='Cap steps for a short real-data check; 0 uses all steps')
    parser.add_argument('--num_thread', type=int, default=8)
    parser.add_argument('--data_root', default='.')
    parser.add_argument('--train_tasks', nargs='+', default=list(TRAIN_TASKS))
    parser.add_argument('--test_tasks', nargs='*', default=list(TEST_TASKS))
    parser.add_argument('--test_dataset_rgbd', nargs='+')
    parser.add_argument('--vdt_train_root', default='')
    parser.add_argument('--vdt_test_root', default='')
    parser.add_argument('--backbone_path', default='vssmsmall_dp03_ckpt_epoch_238.pth')
    parser.add_argument('--model_path', default='', help='Optional SOD warm-start weights')
    parser.add_argument('--resume', default='', help='Full MACL checkpoint, including replay buffers')
    parser.add_argument('--from_scratch', action='store_true', help='Explicitly opt out of RGB pretraining')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--cuda', nargs='?', const='true', choices=['true', 'false', 'True', 'False'])
    parser.add_argument('--local_rank', '--local-rank', type=int, default=int(os.environ.get('LOCAL_RANK', 0)))
    parser.add_argument('--scan_backend', choices=['auto', 'cuda', 'torch'], default='auto')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--validation_frequency', type=int, default=0,
                        help='Validation every N steps and at each epoch end; 0 means epoch end only')
    parser.add_argument('--disable_validation', action='store_true')
    parser.add_argument('--buffer_size_rgb', type=int, default=2048)
    parser.add_argument('--buffer_size_dual', type=int, default=4096)
    parser.add_argument('--stage2_rgb_replay', type=float, default=0.10)
    parser.add_argument('--stage3_rgb_replay', type=float, default=0.10)
    parser.add_argument('--stage3_dual_replay', type=float, default=0.30)
    parser.add_argument('--rq_probability', type=float, default=0.1)
    parser.add_argument('--rq_salient_max_delta', type=float, default=0.05,
                        help='Maximum foreground RQ change as a fraction of channel range')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--check_data', action='store_true')
    config = parser.parse_args(argv)
    if config.cuda is not None:
        config.device = 'cuda' if config.cuda.lower() == 'true' else 'cpu'
    for name in ['epoch', 'epoch_save', 'input_size', 'batch_size', 'test_batch_size', 'accumulation_steps']:
        if getattr(config, name) <= 0:
            parser.error(f'--{name} must be positive')
    if config.input_size < 32 or config.input_size % 32:
        parser.error('--input_size must be a multiple of 32 and at least 32')
    if min(config.stage1_epochs, config.stage2_epochs, config.num_thread, config.validation_frequency) < 0:
        parser.error('Stage lengths, workers and validation frequency cannot be negative')
    if config.stage_patience < 0 or not math.isfinite(config.stage_min_delta) or config.stage_min_delta < 0:
        parser.error('Stage patience and minimum improvement must be nonnegative and finite')
    if config.mode == 'train' and config.stage_patience and (config.disable_validation or not config.test_tasks):
        parser.error('Stage early switching requires validation; use --stage_patience 0 for checks without validation')
    if min(config.max_train_samples, config.max_test_samples, config.max_steps_per_epoch) < 0:
        parser.error('Sample and step limits cannot be negative')
    for name in ['stage2_rgb_replay', 'stage3_rgb_replay', 'stage3_dual_replay', 'rq_probability', 'rq_salient_max_delta']:
        if not 0 <= getattr(config, name) <= 1:
            parser.error(f'--{name} must be between 0 and 1')
    if config.stage3_rgb_replay + config.stage3_dual_replay >= 1 or config.stage2_rgb_replay >= 1:
        parser.error('Replay must leave a positive fraction for current-stage samples')
    if min(config.buffer_size_rgb, config.buffer_size_dual) <= 0:
        parser.error('Replay buffer capacities must be positive')
    if config.mode == 'test' and config.disable_validation:
        parser.error('--mode test cannot be combined with --disable_validation')
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def convert_bn_to_gn(module, num_groups=32):
    if isinstance(module, nn.BatchNorm2d):
        groups = min(num_groups, module.num_features)
        while module.num_features % groups:
            groups -= 1
        result = nn.GroupNorm(groups, module.num_features, eps=module.eps, affine=module.affine)
        if module.affine:
            with torch.no_grad():
                result.weight.copy_(module.weight)
                result.bias.copy_(module.bias)
        return result
    for name, child in module.named_children():
        setattr(module, name, convert_bn_to_gn(child, num_groups))
    return module


def tasks_for_stage(stage, loaders):
    return sorted(set(STAGE_TASKS[stage]).intersection(loaders))


def tests_for_stage(stage, tests):
    """Monitor learned modalities, including earlier tasks to detect forgetting."""
    modality_stages = {
        **dict.fromkeys(['DUTS-TE', 'DUT-OMRON', 'ECSSD', 'HKU-IS', 'PASCAL-S'], 1),
        **dict.fromkeys(['NJU2K', 'NLPR', 'DUT-RGBD', 'SIP', 'STERE', 'VT5000',
                         'VT821', 'VT1000', 'DAVIS', 'DAVSOD', 'FBMS', 'SegTrack-V2', 'VOS'], 2),
        **dict.fromkeys(['RDVS', 'DVisal', 'vidsod_100', 'VDT'], 3),
    }
    return {name: dataset for name, dataset in tests.items() if modality_stages[name] <= stage}


class StageProgress:
    """Epoch-based patience and independent validation bests for each stage."""
    def __init__(self, config):
        self.caps = [min(config.stage1_epochs, config.epoch),
                     min(config.stage2_epochs, max(0, config.epoch - config.stage1_epochs)),
                     max(0, config.epoch - config.stage1_epochs - config.stage2_epochs)]
        self.patience, self.min_delta = config.stage_patience, config.stage_min_delta
        self.stage = next(i + 1 for i, cap in enumerate(self.caps) if cap)
        self.epochs = self.bad_epochs = 0
        self.best_score, self.best_epoch, self.best_checkpoint = None, None, None
        self.history, self.finished = [], False

    def observe(self, score, epoch):
        self.epochs += 1
        improved = False
        if score is not None:
            if not math.isfinite(score):
                raise FloatingPointError('Non-finite stage validation score')
            improved = self.best_score is None or score > self.best_score + self.min_delta
            if improved:
                self.best_score, self.best_epoch, self.bad_epochs = score, epoch + 1, 0
            else:
                self.bad_epochs += 1
        return improved

    def stop_reason(self):
        if self.patience and self.bad_epochs >= self.patience:
            return 'no_improvement'
        if self.epochs >= self.caps[self.stage - 1]:
            return 'epoch_limit'
        return None

    def advance(self):
        self.history.append(dict(stage=self.stage, epochs=self.epochs, best_epoch=self.best_epoch,
                                 best_score=self.best_score, reason=self.stop_reason()))
        following = [s for s in range(self.stage + 1, 4) if self.caps[s - 1]]
        if not following:
            self.finished = True
            return
        self.stage = following[0]
        self.epochs = self.bad_epochs = 0
        self.best_score = self.best_epoch = self.best_checkpoint = None

    def state_dict(self):
        return dict(vars(self))

    def load_state_dict(self, state):
        for key in ['caps', 'patience', 'min_delta']:
            if state[key] != getattr(self, key):
                raise ValueError(f'Resume stage schedule differs: {key}')
        self.__dict__.update(state)


class LossAwareReplayBuffer:
    def __init__(self, capacity, sources):
        self.capacity = capacity
        self.sources = sorted(sources)
        self.items = {source: {} for source in self.sources}
        if self.sources and capacity < len(self.sources):
            raise ValueError('Buffer capacity must cover every selected replay source')

    def __len__(self):
        return sum(len(values) for values in self.items.values())

    def add(self, source, indices, losses):
        bucket = self.items[source]
        for index, loss in zip(indices, losses):
            bucket[int(index)] = max(float(loss), 1e-8)
        limit = self.capacity // max(len(self.sources), 1)
        if len(bucket) > limit:
            self.items[source] = dict(sorted(bucket.items(), key=lambda item: item[1], reverse=True)[:limit])

    def sample(self, source, count, datasets, rng):
        bucket = self.items[source]
        if not bucket:
            raise RuntimeError(f'No replay examples for {source}; train the preceding stage or resume its checkpoint')
        indices = rng.choices(list(bucket), weights=list(bucket.values()), k=count)
        return default_collate([datasets[source][index] for index in indices]), indices

    def state_dict(self):
        return {'capacity': self.capacity, 'sources': self.sources, 'items': self.items}

    def load_state_dict(self, state, dataset_lengths):
        if state['sources'] != self.sources or state['capacity'] != self.capacity:
            raise ValueError('Resume replay sources/capacities differ from this run')
        if set(state['items']) != set(self.sources):
            raise ValueError('Resume replay buckets differ from the selected sources')
        for source, bucket in state['items'].items():
            if source not in dataset_lengths:
                raise ValueError(f'Replay dataset is unavailable: {source}')
            for index in bucket:
                if type(index) is not int or not 0 <= index < dataset_lengths[source]:
                    raise ValueError(f'Replay index out of range for {source}: {index}; '
                                     f'dataset length={dataset_lengths[source]}')
        self.items = state['items']


class ReplaySchedule:
    """Weighted deficit scheduling; counts stay within a sample of target shares."""
    def __init__(self, rgb=0.0, dual=0.0):
        rgb, dual = Fraction(str(rgb)), Fraction(str(dual))
        self.rates = {'current': 1 - rgb - dual, 'rgb': rgb, 'dual': dual}
        self.counts = Counter()
        self.total = 0

    def next(self, batch_size):
        result = Counter()
        for _ in range(batch_size):
            self.total += 1
            source = max((key for key, rate in self.rates.items() if rate > 0),
                         key=lambda key: self.total * self.rates[key] - self.counts[key])
            self.counts[source] += 1
            result[source] += 1
        return result


def slice_batch(batch, count):
    return {key: value[:count] for key, value in batch.items()}


def forward_losses(model, batch, device, stage):
    image = batch['image'].to(device, non_blocking=True)
    label = batch['label'].to(device, non_blocking=True)
    inputs = [batch[key].to(device, non_blocking=True) if key in batch else None
              for key in ('flow', 'depth', 'thermal')]
    auxiliary_count = sum(value is not None for value in inputs)
    if auxiliary_count != stage - 1:
        raise ValueError(f'Stage-{stage} sample requires {stage - 1} auxiliaries, got {auxiliary_count}')
    out, saliency, _ = model(image, *inputs, mode='train', gt=label)
    return (total_loss_suite_optimized(out, label, reduction='none')
            + 0.7 * structure_loss_edge(saliency, label, reduction='none'))


def load_weights(model, path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    state = {key.removeprefix('module.'): value for key, value in state.items()}
    matching = set(model.state_dict()).intersection(state)
    if not matching:
        raise ValueError(f'No matching model weights in {path}')
    result = model.load_state_dict(state, strict=False)
    print(f'Loaded {len(matching)} tensors from {path}; missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}')


def save_checkpoint(model, optimizer, scheduler, scaler, buffers, epoch, global_step,
                    best_sm, config, rank, world_size, schedule=None, data_identity=None,
                    progress=None, filename='last_checkpoint.pth', save_model=True):
    local = {'buffers': [buffer.state_dict() for buffer in buffers],
             'random': random.getstate(), 'numpy': np.random.get_state(),
             'torch': torch.get_rng_state(),
             'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    states = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(states, local)
    else:
        states[0] = local
    if rank == 0:
        raw = model.module if hasattr(model, 'module') else model
        checkpoint = {'model': raw.state_dict(), 'optimizer': optimizer.state_dict(),
                      'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                      'epoch': epoch + 1, 'global_step': global_step, 'best_sm': best_sm,
                      'rank_states': states, 'config': vars(config),
                      'data_identity': data_identity}
        if progress is not None:
            checkpoint['stage_progress'] = progress.state_dict()
        if schedule is not None:
            checkpoint['replay_schedule'] = {'counts': dict(schedule.counts), 'total': schedule.total}
        target = Path(config.save_fold) / filename
        temporary = target.with_suffix('.tmp')
        torch.save(checkpoint, temporary)
        temporary.replace(target)
        if save_model:
            torch.save(raw.state_dict(), Path(config.save_fold) / f'macl_epoch_{epoch + 1}.pth')
    if world_size > 1:
        dist.barrier()


def restore_training_state(checkpoint, model, optimizer, scheduler, scaler, buffers, datasets, rank):
    raw = model.module if hasattr(model, 'module') else model
    raw.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    scaler.load_state_dict(checkpoint['scaler'])
    local = checkpoint['rank_states'][rank]
    lengths = {name: len(dataset) for name, dataset in datasets.items()}
    for buffer, state in zip(buffers, local['buffers']):
        buffer.load_state_dict(state, lengths)
    random.setstate(local['random'])
    np.random.set_state(local['numpy'])
    torch.set_rng_state(local['torch'])
    if local['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(local['cuda'])


def resolve_stage_best(progress, config):
    """A moved resume bundle may keep the best snapshot beside last_checkpoint."""
    name = f'stage{progress.stage}_best.pth'
    candidates = [Path(config.save_fold) / name]
    if config.resume:
        candidates.append(Path(config.resume).resolve().parent / name)
    if progress.best_checkpoint:
        candidates.append(Path(progress.best_checkpoint))
    for path in candidates:
        if path.is_file():
            saved = torch.load(path, map_location='cpu', weights_only=False)
            metadata = saved.get('stage_progress', {})
            if (metadata.get('stage') == progress.stage and metadata.get('best_epoch') == progress.best_epoch
                    and metadata.get('best_score') == progress.best_score):
                return path, saved
    raise FileNotFoundError(f'Missing matching {name}; resume requires last_checkpoint.pth and the current stage best snapshot')


def finish_stage(progress, model, optimizer, scheduler, scaler, buffers, datasets,
                 data_identity, config, rank):
    if progress.best_score is not None:
        path, best_snapshot = resolve_stage_best(progress, config)
        validate_training_data_identity(best_snapshot.get('data_identity'), data_identity)
        restore_training_state(best_snapshot, model, optimizer, scheduler, scaler, buffers, datasets, rank)
        del best_snapshot
        if rank == 0:
            print(f'Stage-{progress.stage} ends ({progress.stop_reason()}); restored best epoch '
                  f'{progress.best_epoch}, S-measure={progress.best_score:.6f}: {path}')
            raw = model.module if hasattr(model, 'module') else model
            torch.save(raw.state_dict(), Path(config.save_fold) / 'overall_best.pth')
    progress.advance()
    if rank == 0:
        with (Path(config.save_fold) / 'stage_summary.json').open('w', encoding='utf-8') as stream:
            json.dump(progress.history, stream, indent=2)


def run(config):
    distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    if config.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this Python environment; use --device cpu for checks')
    if config.device == 'cuda':
        torch.cuda.set_device(config.local_rank)
    device = torch.device(config.device, config.local_rank) if config.device == 'cuda' else torch.device('cpu')
    if distributed:
        backend = 'nccl' if device.type == 'cuda' and dist.is_nccl_available() else 'gloo'
        if os.name == 'nt' and backend == 'gloo':
            try:
                dist.ProcessGroupGloo.create_device(hostname='127.0.0.1')
            except RuntimeError as exc:
                raise RuntimeError('This Windows PyTorch build has no usable Gloo transport. '
                                   'Run torchrun in Linux/WSL with a working PyTorch build; '
                                   'single-GPU training remains available here.') from exc
        dist.init_process_group(backend=backend)
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    set_seed(config.seed)
    loaders, tests = setup_all_dataloaders_for_joint_training(config, rank, world_size)
    data_identity = build_training_data_identity(loaders, config) if config.mode == 'train' else None
    if distributed and data_identity is not None:
        identities = [None] * world_size
        dist.all_gather_object(identities, data_identity)
        for peer_identity in identities:
            validate_training_data_identity(identities[0], peer_identity)
    if config.check_data:
        if rank == 0:
            print(f'Data checks passed: {list(loaders)}, tests={list(tests)}')
        return
    if config.mode == 'train':
        progress = StageProgress(config)
        active_stages = {s + 1 for s, cap in enumerate(progress.caps) if cap}
        for stage in active_stages:
            if not tasks_for_stage(stage, loaders):
                raise ValueError(f'No selected tasks for Stage-{stage}: choose from {STAGE_TASKS[stage]}')
            if config.stage_patience and not tests_for_stage(stage, tests):
                raise ValueError(f'No validation datasets for learned modalities in Stage-{stage}')
        if config.resume == '' and (config.stage1_epochs == 0 or (3 in active_stages and config.stage2_epochs == 0)):
            raise ValueError('Later stages require previous-stage buffers; run all preceding stages or --resume')
    if config.mode == 'test' and not (config.model_path or config.resume):
        raise ValueError('--mode test requires --model_path or --resume')
    if config.mode == 'test' and not tests:
        raise ValueError('--mode test requires at least one --test_tasks dataset')
    backbone_path = None if config.from_scratch or config.model_path or config.resume else config.backbone_path
    for path in [backbone_path, config.model_path, config.resume]:
        if path and not Path(path).is_file():
            raise FileNotFoundError(f'Checkpoint not found: {path}. Set its path, or explicitly use --from_scratch.')
    if config.resume:
        checkpoint = torch.load(config.resume, map_location='cpu', weights_only=False)
        if config.mode == 'train':
            # Reject stale replay mappings before allocating/loading the model.
            validate_training_data_identity(checkpoint.get('data_identity'), data_identity)
            if 'stage_progress' not in checkpoint:
                raise ValueError('Resume checkpoint has no stage progress; use --model_path for a weights-only warm start')
    os.environ['SAMBA_SCAN_BACKEND'] = config.scan_backend
    from models.SalMamba_tri import Model
    model = convert_bn_to_gn(Model(backbone_pretrained=backbone_path,
                                  rq_probability=config.rq_probability,
                                  rq_salient_max_delta=config.rq_salient_max_delta)).to(device)
    if config.model_path:
        load_weights(model, config.model_path)
    if config.resume:
        model.load_state_dict(checkpoint['model'])
    config.save_fold = str(Path(config.save_fold).resolve() / 'MACL_Training')
    if rank == 0:
        Path(config.save_fold).mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    if config.mode == 'test':
        evaluate_all_tests_and_save_best(model, tests, 0, config.save_fold, config)
        return
    if distributed:
        model = DDP(model, device_ids=[config.local_rank] if device.type == 'cuda' else None,
                    find_unused_parameters=True, broadcast_buffers=False)
    raw = model.module if distributed else model
    optimizer = torch.optim.AdamW(group_weight([], raw, nn.BatchNorm2d, config.lr),
                                  lr=config.lr, betas=(0.85, 0.999), weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epoch, eta_min=1e-5)
    scaler = GradScaler(enabled=device.type == 'cuda')
    rgb_buffer = LossAwareReplayBuffer(config.buffer_size_rgb, tasks_for_stage(1, loaders))
    dual_buffer = LossAwareReplayBuffer(config.buffer_size_dual, tasks_for_stage(2, loaders))
    datasets = {name: loader.dataset for name, loader in loaders.items()}
    start_epoch, global_step, best_sm = 0, 0, -1.0
    if config.resume:
        if len(checkpoint['rank_states']) != world_size:
            raise ValueError('Resume requires the same world size to preserve replay state')
        for key in ['data_root', 'train_tasks', 'stage1_epochs', 'stage2_epochs', 'batch_size',
                    'input_size', 'epoch', 'accumulation_steps', 'stage2_rgb_replay',
                    'stage3_rgb_replay', 'stage3_dual_replay', 'rq_probability', 'rq_salient_max_delta',
                    'max_train_samples', 'max_test_samples', 'max_steps_per_epoch',
                    'stage_patience', 'stage_min_delta', 'test_tasks', 'test_dataset_rgbd',
                    'disable_validation', 'validation_frequency']:
            if checkpoint['config'].get(key) != getattr(config, key):
                raise ValueError(f'Resume configuration differs: {key}')
        restore_training_state(checkpoint, model, optimizer, scheduler, scaler,
                               [rgb_buffer, dual_buffer], datasets, rank)
        progress.load_state_dict(checkpoint['stage_progress'])
        start_epoch, global_step, best_sm = checkpoint['epoch'], checkpoint['global_step'], checkpoint['best_sm']
        restored_schedule = checkpoint.get('replay_schedule')
        del checkpoint
    else:
        restored_schedule = None
    config._best_validation_sm = best_sm
    if config.resume and progress.best_score is not None and not progress.finished:
        # Fail before training if a required best snapshot was not copied with a resume bundle.
        best_path, best_snapshot = resolve_stage_best(progress, config)
        validate_training_data_identity(best_snapshot.get('data_identity'), data_identity)
        target = Path(config.save_fold) / f'stage{progress.stage}_best.pth'
        if rank == 0 and best_path.resolve() != target.resolve():
            temporary = target.with_suffix('.tmp')
            shutil.copyfile(best_path, temporary)
            temporary.replace(target)
            torch.save(best_snapshot['model'], target.with_name(f'stage{progress.stage}_best_model.pth'))
        if distributed:
            dist.barrier()
        progress.best_checkpoint = str(target)
        del best_snapshot
    if config.resume and not progress.finished and progress.stop_reason() is not None:
        # A best snapshot can have been saved on the epoch that hit the cap.
        finish_stage(progress, model, optimizer, scheduler, scaler, [rgb_buffer, dual_buffer],
                     datasets, data_identity, config, rank)
        restored_schedule = None
        save_checkpoint(model, optimizer, scheduler, scaler, [rgb_buffer, dual_buffer], start_epoch - 1,
                        global_step, best_sm, config, rank, world_size,
                        data_identity=data_identity, progress=progress, save_model=False)
    previous_stage = None
    schedule = None
    for epoch in range(start_epoch, config.epoch):
        if progress.finished:
            break
        stage = progress.stage
        stage_tests = tests_for_stage(stage, tests)
        config._validation_stage = stage
        names = tasks_for_stage(stage, loaders)
        if stage != previous_stage:
            config._best_validation_sm = progress.best_score if progress.best_score is not None else -1.0
            best_sm = config._best_validation_sm
            if progress.epochs == 0:
                # Each new modality expansion gets its own learning-rate schedule.
                for group in optimizer.param_groups:
                    group['lr'] = group['initial_lr'] = config.lr
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=progress.caps[stage - 1], eta_min=1e-5)
            rgb_rate = config.stage2_rgb_replay if stage == 2 else config.stage3_rgb_replay
            if stage >= 2 and rgb_rate > 0 and not len(rgb_buffer):
                raise RuntimeError('RGB replay buffer is empty at the stage transition')
            if stage == 3 and config.stage3_dual_replay > 0 and not len(dual_buffer):
                raise RuntimeError('Dual replay buffer is empty at the stage transition')
            schedule = ReplaySchedule(config.stage2_rgb_replay if stage == 2 else config.stage3_rgb_replay if stage == 3 else 0,
                                      config.stage3_dual_replay if stage == 3 else 0)
            if restored_schedule and progress.epochs > 0:
                schedule.counts.update(restored_schedule['counts'])
                schedule.total = restored_schedule['total']
            restored_schedule = None
            previous_stage = stage
        for name in names:
            loaders[name].sampler.set_epoch(epoch)
        iterators = {name: iter(loaders[name]) for name in names}
        steps = max(len(loaders[name]) for name in names) * len(names)
        if config.max_steps_per_epoch:
            steps = min(steps, config.max_steps_per_epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_counts, loss_sum, last_validation, epoch_score = Counter(), 0.0, -1, None
        if rank == 0:
            print(f'Epoch {epoch + 1}/{config.epoch} Stage-{stage}: {names}; buffers RGB={len(rgb_buffer)}, dual={len(dual_buffer)}')
        for step in tqdm(range(steps), disable=rank != 0):
            source_rng = random.Random(config.seed + global_step)
            task = source_rng.choice(names) if stage >= 2 else names[step % len(names)]
            counts = schedule.next(config.batch_size)
            pieces = []
            if counts['current']:
                try:
                    batch = next(iterators[task])
                except StopIteration:
                    iterators[task] = iter(loaders[task])
                    batch = next(iterators[task])
                batch = slice_batch(batch, counts['current'])
                pieces.append(('current', task, batch, batch['sample_index'].tolist(), stage))
            for kind, buffer, replay_stage in [('rgb', rgb_buffer, 1), ('dual', dual_buffer, 2)]:
                if counts[kind]:
                    available = [source for source in buffer.sources if buffer.items[source]]
                    if not available:
                        raise RuntimeError(f'{kind} replay buffer is empty')
                    source = source_rng.choice(available)
                    batch, indices = buffer.sample(source, counts[kind], datasets, random)
                    pieces.append((kind, source, batch, indices, replay_stage))
            window_start = step // config.accumulation_steps * config.accumulation_steps
            window_steps = min(config.accumulation_steps, steps - window_start)
            for kind, source, batch, indices, sample_stage in pieces:
                with autocast(enabled=device.type == 'cuda'):
                    losses = forward_losses(model, batch, device, sample_stage)
                    loss = losses.sum() / (window_steps * config.batch_size)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite loss: Stage-{stage}, {source}')
                scaler.scale(loss).backward()
                priorities = losses.detach().float().cpu().tolist()
                if kind == 'rgb' or (kind == 'current' and stage == 1):
                    rgb_buffer.add(source, indices, priorities)
                elif kind == 'dual' or (kind == 'current' and stage == 2):
                    dual_buffer.add(source, indices, priorities)
                epoch_counts[kind] += len(indices)
                loss_sum += sum(priorities)
            if (step + 1) % config.accumulation_steps == 0 or step + 1 == steps:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if stage_tests and not config.disable_validation and config.validation_frequency > 0 and global_step % config.validation_frequency == 0:
                epoch_score = evaluate_all_tests_and_save_best(model, stage_tests, epoch, config.save_fold, config)
                last_validation = global_step
        if stage_tests and not config.disable_validation and last_validation != global_step:
            epoch_score = evaluate_all_tests_and_save_best(model, stage_tests, epoch, config.save_fold, config)
        improved = progress.observe(epoch_score, epoch)
        best_sm = progress.best_score if progress.best_score is not None else -1.0
        if improved:
            progress.best_checkpoint = str(Path(config.save_fold) / f'stage{stage}_best.pth')
        scheduler.step()
        row = {'epoch': epoch + 1, 'stage': stage, 'stage_epoch': progress.epochs,
               'avg_loss': loss_sum / sum(epoch_counts.values()), 'validation_sm': epoch_score,
               'stage_best_sm': progress.best_score, 'stage_best_epoch': progress.best_epoch,
               'no_improvement_epochs': progress.bad_epochs, 'samples': dict(epoch_counts),
               'rgb_buffer': len(rgb_buffer), 'dual_buffer': len(dual_buffer)}
        if rank == 0:
            print(row)
            with (Path(config.save_fold) / 'training_log_macl.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row) + '\n')
        if improved:
            save_checkpoint(model, optimizer, scheduler, scaler, [rgb_buffer, dual_buffer], epoch,
                            global_step, best_sm, config, rank, world_size, schedule=schedule,
                            data_identity=data_identity, progress=progress,
                            filename=f'stage{stage}_best.pth', save_model=False)
            if rank == 0:
                torch.save(raw.state_dict(), Path(config.save_fold) / f'stage{stage}_best_model.pth')
        boundary = progress.stop_reason() is not None
        if boundary:
            finish_stage(progress, model, optimizer, scheduler, scaler, [rgb_buffer, dual_buffer],
                         datasets, data_identity, config, rank)
            schedule = None
        # Persist patience every epoch, including a just-completed transition.
        save_checkpoint(model, optimizer, scheduler, scaler, [rgb_buffer, dual_buffer], epoch,
                        global_step, best_sm, config, rank, world_size, schedule=schedule,
                        data_identity=data_identity, progress=progress,
                        save_model=boundary or (epoch + 1) % config.epoch_save == 0)
    if rank == 0:
        print(f'MACL training finished; best S-measure={best_sm:.4f}')


def main(argv=None):
    config = parse_args(argv)
    try:
        run(config)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
