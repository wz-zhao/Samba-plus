import argparse
import copy
import os
import random
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "dataset"))
sys.path.insert(0, os.path.join(ROOT, "transform"))

import transform_dvsod
import transform_rgbd
import transform_rgbt
import transform_single
import transform_vsod
from dataset.dvsod_dataset import Dataset as DVSODDataset
from dataset.rgb_dataset import Dataset as RGBDataset
from dataset.rgbd_dataset import Dataset as RGBDDataset
from dataset.rgbt_dataset import Dataset as RGBTDataset
from dataset.vdt_dataset import Data as VDTDataset
from dataset.vsod_dataset import Dataset as VSODDataset
from loss import structure_loss_edge, total_loss_suite_optimized
from models.SalMamba_tri import Model
from utils.init_func import group_weight

try:
    from val_join_ddp import evaluate_all_tests_and_save_best
except Exception:
    evaluate_all_tests_and_save_best = None


p = OrderedDict()
p["lr"] = 1e-4
p["wd"] = 0.01


def parse_args():
    parser = argparse.ArgumentParser("MACL progressive DDP training for Samba+")
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--stage1_epochs", type=int, default=80)
    parser.add_argument("--stage2_epochs", type=int, default=180)
    parser.add_argument("--epoch_save", type=int, default=5)
    parser.add_argument("--save_fold", type=str, default="./checkpoints")
    parser.add_argument("--input_size", type=int, default=448)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--num_thread", type=int, default=8)
    parser.add_argument("--model_path", type=str, default="./Samba++.pth")
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--validation_frequency", type=int, default=1172)
    parser.add_argument("--disable_validation", action="store_true")
    parser.add_argument("--buffer_size_rgb", type=int, default=2048)
    parser.add_argument("--buffer_size_dual", type=int, default=4096)
    parser.add_argument("--stage2_rgb_replay", type=float, default=0.10)
    parser.add_argument("--stage3_rgb_replay", type=float, default=0.10)
    parser.add_argument("--stage3_dual_replay", type=float, default=0.30)
    parser.add_argument("--vdt_train_root", type=str, default="/home/user0/BRL/fabian/BRL/zhaowenzhuo/Sigma/VDT-2048 dataset/Train/")
    parser.add_argument("--vdt_test_root", type=str, default="/home/user0/BRL/fabian/BRL/zhaowenzhuo/Sigma/VDT-2048 dataset/Test/")
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def convert_bn_to_gn(module, num_groups=32):
    if isinstance(module, torch.nn.BatchNorm2d):
        groups = min(num_groups, module.num_features)
        while module.num_features % groups != 0 and groups > 1:
            groups -= 1
        return torch.nn.GroupNorm(groups, module.num_features)
    for name, child in module.named_children():
        setattr(module, name, convert_bn_to_gn(child, num_groups))
    return module


def strip_module_prefix(state_dict):
    if all(k.startswith("module.") for k in state_dict.keys()):
        return OrderedDict((k.replace("module.", "", 1), v) for k, v in state_dict.items())
    return state_dict


class LossAwareReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.items = []

    def __len__(self):
        return len(self.items)

    def add_batch(self, batch, loss_value, source):
        if self.capacity <= 0:
            return
        batch_size = batch["image"].shape[0]
        for idx in range(batch_size):
            sample = {"source": source}
            for key, value in batch.items():
                if torch.is_tensor(value):
                    sample[key] = value[idx].detach().cpu()
            self.items.append((float(loss_value), sample))
        if len(self.items) > self.capacity:
            self.items.sort(key=lambda x: x[0], reverse=True)
            self.items = self.items[: self.capacity]

    def sample(self, batch_size):
        if not self.items:
            return None
        hard_pool = self.items[: max(batch_size, len(self.items) // 2)]
        chosen = random.choices(hard_pool, k=batch_size)
        samples = [copy.deepcopy(item[1]) for item in chosen]
        keys = set().union(*(sample.keys() for sample in samples))
        batch = {}
        for key in keys:
            values = [sample.get(key) for sample in samples]
            if all(torch.is_tensor(v) for v in values):
                batch[key] = torch.stack(values, dim=0)
        return batch


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


def safe_dataset(rank, name, factory):
    try:
        dataset = factory()
        if len(dataset) == 0:
            if rank == 0:
                print(f"Skipping empty dataset: {name}")
            return None
        return dataset
    except Exception as exc:
        if rank == 0:
            print(f"Skipping dataset {name}: {exc}")
        return None


def make_loader(dataset, config, rank, world_size):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_thread,
        drop_last=True,
        pin_memory=True,
        sampler=sampler,
    )
    return loader


def setup_macl_dataloaders(config, rank, world_size):
    rgb_tr, rgb_te, rgbd_tr, rgbd_te, rgbt_tr, rgbt_te, vsod_tr, dvsod_tr = build_transforms(config)
    datasets = {
        "DUTS-TR": safe_dataset(rank, "DUTS-TR", lambda: RGBDataset(["DUTS-TR"], transform=rgb_tr, mode="train")),
        "train_DUT": safe_dataset(rank, "train_DUT", lambda: RGBDDataset(["train_DUT"], transform=rgbd_tr, mode="train")),
        "VT_train": safe_dataset(rank, "VT_train", lambda: RGBTDataset(["VT_train"], transform=rgbt_tr, mode="train")),
        "VSOD_train": safe_dataset(rank, "VSOD_train", lambda: VSODDataset(["DAVIS", "DAVSOD", "FBMS"], transform=vsod_tr, mode="train")),
        "DVSOD_train": safe_dataset(rank, "DVSOD_train", lambda: DVSODDataset(["RDVS", "DVisal", "vidsod_100"], transform=dvsod_tr, mode="train")),
        "VDT": safe_dataset(rank, "VDT", lambda: VDTDataset(config.vdt_train_root, mode="train")),
    }
    loaders = {name: make_loader(ds, config, rank, world_size) for name, ds in datasets.items() if ds is not None}

    tests = {
        "DUTS-TE": safe_dataset(rank, "DUTS-TE", lambda: RGBDataset(["DUTS-TE"], transform=rgb_te, mode="test")),
        "DUT-OMRON": safe_dataset(rank, "DUT-OMRON", lambda: RGBDataset(["DUT-OMRON"], transform=rgb_te, mode="test")),
        "NJU2K": safe_dataset(rank, "NJU2K", lambda: RGBDDataset(["NJU2K"], transform=rgbd_te, mode="test")),
        "NLPR": safe_dataset(rank, "NLPR", lambda: RGBDDataset(["NLPR"], transform=rgbd_te, mode="test")),
        "VT5000": safe_dataset(rank, "VT5000", lambda: RGBTDataset(["VT5000"], transform=rgbt_te, mode="test")),
    }
    tests = {name: ds for name, ds in tests.items() if ds is not None}
    return loaders, tests


def stage_for_epoch(epoch, config):
    if epoch < config.stage1_epochs:
        return 1
    if epoch < config.stage1_epochs + config.stage2_epochs:
        return 2
    return 3


def tasks_for_stage(stage, loaders):
    rgb = ["DUTS-TR"]
    dual = ["train_DUT", "VT_train", "VSOD_train"]
    tri = ["DVSOD_train", "VDT"]
    if stage == 1:
        names = rgb
    elif stage == 2:
        names = dual
    else:
        names = tri
    return [name for name in names if name in loaders]


def move_batch_to_device(batch, device):
    image = batch["image"].to(device, non_blocking=True)
    label = batch["label"].to(device, non_blocking=True)
    flow = batch.get("flow")
    depth = batch.get("depth")
    thermal = batch.get("thermal")
    flow = flow.to(device, non_blocking=True) if torch.is_tensor(flow) else None
    depth = depth.to(device, non_blocking=True) if torch.is_tensor(depth) else None
    thermal = thermal.to(device, non_blocking=True) if torch.is_tensor(thermal) else None
    return image, label, flow, depth, thermal


def forward_loss(model, batch, device):
    image, label, flow, depth, thermal = move_batch_to_device(batch, device)
    out, saliency, _ = model(image, flow, depth, thermal, mode="train", gt=label)
    loss = total_loss_suite_optimized(out, label) + 0.7 * structure_loss_edge(saliency, label)
    return loss


def choose_batch(stage, task_name, data_batch, rgb_buffer, dual_buffer, config):
    replay_source = "current"
    if stage == 2 and len(rgb_buffer) > 0 and random.random() < config.stage2_rgb_replay:
        replay = rgb_buffer.sample(config.batch_size)
        if replay is not None:
            return replay, "rgb_replay"
    if stage == 3:
        replay_roll = random.random()
        if len(rgb_buffer) > 0 and replay_roll < config.stage3_rgb_replay:
            replay = rgb_buffer.sample(config.batch_size)
            if replay is not None:
                return replay, "rgb_replay"
        if len(dual_buffer) > 0 and replay_roll < config.stage3_rgb_replay + config.stage3_dual_replay:
            replay = dual_buffer.sample(config.batch_size)
            if replay is not None:
                return replay, "dual_replay"
    return data_batch, replay_source if replay_source != "current" else task_name


def main():
    config = parse_args()
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = config.local_rank
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if rank == 0:
        config.save_fold = os.path.join(config.save_fold, "MACL_Training")
        os.makedirs(config.save_fold, exist_ok=True)

    set_seed(1024)
    model = convert_bn_to_gn(Model()).to(device)
    if config.model_path:
        try:
            pretrained_dict = strip_module_prefix(torch.load(config.model_path, map_location="cpu"))
            model.load_state_dict(pretrained_dict, strict=False)
            if rank == 0:
                print(f"Loaded pretrained model from {config.model_path}")
        except Exception as exc:
            if rank == 0:
                print(f"Could not load pretrained model from {config.model_path}: {exc}")

    dist.barrier()
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True, broadcast_buffers=False)

    params_list = group_weight([], model.module, nn.BatchNorm2d, p["lr"])
    optimizer = torch.optim.AdamW(params_list, lr=p["lr"], betas=(0.85, 0.999), weight_decay=p["wd"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
    scaler = GradScaler()

    loaders, test_datasets = setup_macl_dataloaders(config, rank, world_size)
    if not loaders:
        if rank == 0:
            print("No training dataloaders found. Exiting.")
        dist.destroy_process_group()
        return

    iterators = {name: iter(loader) for name, loader in loaders.items()}
    rgb_buffer = LossAwareReplayBuffer(config.buffer_size_rgb)
    dual_buffer = LossAwareReplayBuffer(config.buffer_size_dual)
    best_overall_sm = -1.0
    best_epoch_for_overall = -1
    global_step = 0
    epoch_log = []

    for epoch in range(config.epoch):
        stage = stage_for_epoch(epoch, config)
        task_names = tasks_for_stage(stage, loaders)
        if not task_names:
            if rank == 0:
                print(f"Epoch {epoch + 1}: no loaders available for stage {stage}; skipping.")
            continue

        for loader in loaders.values():
            loader.sampler.set_epoch(epoch)

        max_loader_len = max(len(loaders[name]) for name in task_names)
        steps_per_epoch = max_loader_len * len(task_names)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss_epoch = 0.0

        if rank == 0:
            print(f"\n--- Epoch {epoch + 1}/{config.epoch} | MACL Stage-{stage} | tasks: {task_names} ---")
            print(f"Replay buffers: RGB={len(rgb_buffer)}, Dual={len(dual_buffer)}")

        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}/{config.epoch}", unit="step", disable=(rank != 0))
        for step in pbar:
            if stage == 2:
                task_name = random.choice(task_names)
            else:
                task_name = task_names[step % len(task_names)]
            try:
                data_batch = next(iterators[task_name])
            except StopIteration:
                iterators[task_name] = iter(loaders[task_name])
                data_batch = next(iterators[task_name])

            train_batch, source = choose_batch(stage, task_name, data_batch, rgb_buffer, dual_buffer, config)
            is_update_step = (step + 1) % config.accumulation_steps == 0

            sync_context = model.no_sync() if not is_update_step else torch.enable_grad()
            with sync_context:
                with autocast():
                    loss = forward_loss(model, train_batch, device) / config.accumulation_steps
                scaler.scale(loss).backward()

            step_loss = loss.item() * config.accumulation_steps
            running_loss_epoch += step_loss

            if source == task_name:
                if stage == 1:
                    rgb_buffer.add_batch(data_batch, step_loss, task_name)
                elif stage == 2:
                    dual_buffer.add_batch(data_batch, step_loss, task_name)

            if is_update_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if rank == 0:
                pbar.set_postfix(OrderedDict(Stage=stage, Task=source, Loss=f"{step_loss:.4f}", LR=f"{optimizer.param_groups[0]['lr']:.6f}"))

            global_step += 1
            should_validate = (
                not config.disable_validation
                and evaluate_all_tests_and_save_best is not None
                and test_datasets
                and global_step > 0
                and global_step % config.validation_frequency == 0
            )
            if should_validate:
                scheduler.step()
                dist.barrier()
                current_sm = evaluate_all_tests_and_save_best(model, test_datasets, epoch, config.save_fold, config)
                if rank == 0 and current_sm > best_overall_sm:
                    best_overall_sm = current_sm
                    best_epoch_for_overall = epoch + 1
                    best_model_path = os.path.join(config.save_fold, "best_model_macl.pth")
                    torch.save(model.module.state_dict(), best_model_path)
                    print(f"New best model saved to {best_model_path}, step={global_step}, S-measure={best_overall_sm:.4f}")
                dist.barrier()
                model.train()

        if steps_per_epoch % config.accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if rank == 0:
            avg_loss = running_loss_epoch / max(1, steps_per_epoch)
            epoch_log.append({"epoch": epoch + 1, "stage": stage, "avg_loss": avg_loss, "rgb_buffer": len(rgb_buffer), "dual_buffer": len(dual_buffer)})
            print(f"Epoch {epoch + 1} finished. Average loss: {avg_loss:.4f}")
            with open(os.path.join(config.save_fold, "training_log_macl.txt"), "w") as f:
                for item in epoch_log:
                    f.write(str(item) + "\n")
            if (epoch + 1) % config.epoch_save == 0:
                ckpt_path = os.path.join(config.save_fold, f"macl_epoch_{epoch + 1}.pth")
                torch.save(model.module.state_dict(), ckpt_path)

    if rank == 0:
        print("\n--- MACL Training Finished ---")
        print(f"Best Overall S-measure: {best_overall_sm:.4f} at Epoch: {best_epoch_for_overall}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
