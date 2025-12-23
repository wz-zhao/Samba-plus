import argparse
import os
import numpy as np
import random
from collections import OrderedDict
from datetime import datetime
from tqdm import tqdm
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils import data
from torch.cuda.amp import autocast, GradScaler
from loss import structure_loss_edge,total_loss_suite_optimized
from itertools import combinations
from dvsod_dataset import Dataset as dvsod_Dataset
from torchvision import transforms
import transform_dvsod
from models.SalMamba_tri import Model
from thop import profile
import IOU
from Smeasure import S_object, S_region
from utils.init_func import group_weight, group_weight_new
from get_dataloaders_ddp import setup_all_dataloaders_for_joint_training
from val_join_ddp import evaluate_all_tests_and_save_best


p = OrderedDict()
p['lr'] = 1e-4
p['wd'] = 0.01
p['momentum'] = 0.90
showEvery = 500
from contextlib import nullcontext

CE = torch.nn.BCEWithLogitsLoss(reduction='mean')
IOU = IOU.IOU(size_average=True)


best_overall_sm = -1.0
best_epoch_for_overall = -1
global_test_results_log = []

parser = argparse.ArgumentParser()

parser.add_argument('--cuda', type=bool, default=True, help='cuda')
parser.add_argument('--epoch', type=int, default=500, help='total epoch')
parser.add_argument('--epoch_save', type=int, default=5, help='save_fre')
parser.add_argument('--save_fold', type=str, default='./checkpoints', help='save_path')
parser.add_argument('--input_size', type=int, default=448, help='input_size')
parser.add_argument('--batch_size', type=int, default=3, help='batch_size') 
parser.add_argument('--num_thread', type=int, default=8, help='cpu')
parser.add_argument('--model_path', type=str, default='./Samba++.pth', help='model_checkpoint')

parser.add_argument(
    "--local_rank",
    type=int,
    default=0
)

parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='mode')
config = parser.parse_args()



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    rank = 0
    world_size = 1
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    if rank == 0:
        config.save_fold = os.path.join(config.save_fold, 'Zhao/Joint_Training')
        os.makedirs(config.save_fold, exist_ok=True)

    fixed_model_init_seed = 1024
    set_seed(fixed_model_init_seed)
    model = Model()
    def convert_bn_to_gn(module, num_groups=32):
        if isinstance(module, torch.nn.BatchNorm2d):
            return torch.nn.GroupNorm(num_groups, module.num_features)
        for name, child in module.named_children():
            setattr(module, name, convert_bn_to_gn(child, num_groups))
        return module

    model = convert_bn_to_gn(model)
    model = model.to(device)

    if config.model_path:
        print(f"Loading pre-trained model from: {config.model_path}")
        try:
            pretrained_dict = torch.load(config.model_path, map_location='cpu')
            if all(k.startswith('module.') for k in pretrained_dict.keys()):
                new_state_dict = OrderedDict()
                for k, v in pretrained_dict.items():
                    new_state_dict[k.replace('module.', '')] = v
                pretrained_dict = new_state_dict

            incompatible = model.load_state_dict(pretrained_dict, strict=False)
            print("Pre-trained model loaded successfully (with possible mismatches).")
        except FileNotFoundError:
            print(f"Error: Pre-trained model file not found at {config.model_path}. Starting training from scratch.")
        except Exception as e:
            print(f"Error loading pre-trained model: {e}. Starting training from scratch.")


    input1, input2, input3 ,input4,GT= torch.randn(1, 3, 448, 448), torch.randn(1, 3, 448, 448), torch.randn(1, 3, 448, 448), torch.zeros(1, 3, 448, 448), torch.randn(1, 1, 448, 448)
    input1, input2, input3, input4 ,GT= input1.to(device), input2.to(device), input3.to(device), input4.to(device), GT.to(device)
    mode = None
    input0 = None
    if rank == 0:
        flops, params = profile(model, inputs=(input1, input0, input1,input1,mode,GT))
        print(f"FLOPs: {flops}, Params: {params}")


    params_list = []
    params_list = group_weight(params_list, model, nn.BatchNorm2d, p['lr'])

    optimizer = torch.optim.AdamW(params_list, lr=p['lr'], betas=(0.85, 0.999), weight_decay=p['wd'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=10,
        eta_min=1e-5,
        last_epoch=-1
    )

    scaler = GradScaler()
    optimizer.zero_grad()

    all_train_loaders, all_test_datasets = setup_all_dataloaders_for_joint_training(
        config, 0, world_size
    )



    train_iterators = {name: iter(dl) for name, dl in all_train_loaders.items()}

    if not all_train_loaders:
        print("No training data loaders found. Exiting.")
        return

    loader_lengths = [len(dl) for dl in all_train_loaders.values()] 
    num_tasks = len(all_train_loaders)
    max_loader_len = max(loader_lengths) if loader_lengths else 0
    steps_per_epoch = num_tasks * max_loader_len
    accumulation_steps = 4

    if rank == 0:
        print("\n--- Training Configuration (Single GPU) ---")
        print(f"Longest dataloader length: {max_loader_len} batches")
        print(f"Number of data modalities/tasks: {num_tasks}")
        print(f"Total steps per epoch: {steps_per_epoch}")
        print("---------------------------------------------")

    task_names = sorted(list(all_train_loaders.keys()))
    epoch_results_log = []
    best_overall_sm = 0.0
    global_step = 0
    validation_frequency = 2000   


    for epoch in range(config.epoch):
        model.train()
        running_loss_epoch = 0.0

        for loader in all_train_loaders.values():
            if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(epoch)

        pbar_desc = f"Epoch {epoch+1}/{config.epoch}"
        pbar = tqdm(range(steps_per_epoch), desc=pbar_desc, unit="step", disable=(rank != 0))

        optimizer.zero_grad()

        for step in pbar:
            current_task_name = task_names[step % len(task_names)]
            current_iterator = train_iterators[current_task_name]

            try:
                data_batch = next(current_iterator)
            except StopIteration:
                if hasattr(all_train_loaders[current_task_name], "sampler") and hasattr(all_train_loaders[current_task_name].sampler, "set_epoch"):
                    all_train_loaders[current_task_name].sampler.set_epoch(epoch)
                new_iterator = iter(all_train_loaders[current_task_name])
                train_iterators[current_task_name] = new_iterator
                data_batch = next(new_iterator)

            image = data_batch['image'].to(device, non_blocking=True)
            label = data_batch['label'].to(device, non_blocking=True)

            modal_flow_input = data_batch.get('flow')
            modal_depth_input = data_batch.get('depth')
            modal_thermal_input = data_batch.get('thermal')

            if modal_flow_input is not None:
                modal_flow_input = modal_flow_input.to(device, non_blocking=True)
            if modal_depth_input is not None:
                modal_depth_input = modal_depth_input.to(device, non_blocking=True)
            if modal_thermal_input is not None:
                modal_thermal_input = modal_thermal_input.to(device, non_blocking=True)

            with autocast():
                out, saliency, aux_preds = model(
                    image, modal_flow_input, modal_depth_input, modal_thermal_input,
                    mode='train', gt=label, task='None'
                )

                main_loss = total_loss_suite_optimized(out, label) + 0.7 * structure_loss_edge(saliency, label)
                loss_kd = 0.0
                lambda_kd = 0.5
                loss = main_loss + lambda_kd * loss_kd
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if ((step + 1) % accumulation_steps == 0) or (step + 1 == steps_per_epoch):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            step_loss = loss.item() * accumulation_steps  
            running_loss_epoch += step_loss

            if rank == 0:
                pbar.set_postfix(OrderedDict(
                    Loss=f"{step_loss:.4f}",
                    LR=f"{optimizer.param_groups[0]['lr']:.6f}",
                ))

            global_step += 1


            if global_step > 0 and global_step % validation_frequency == 0:
                scheduler.step()
                if rank == 0:
                    print(f"\n--- Reached step {global_step}. Starting validation... ---")

                current_sm = evaluate_all_tests_and_save_best(model, all_test_datasets, epoch, config.save_fold, config)

                if rank == 0 and current_sm > best_overall_sm:
                    best_overall_sm = current_sm
                    best_epoch_for_overall = epoch + 1
                    best_model_path = os.path.join(config.save_fold, f'best_model_step_0000.pth')
                    torch.save(model.state_dict(), best_model_path)
                    print(f"🎉 New best model saved to {best_model_path}, Step: {global_step}, S-measure: {best_overall_sm:.4f}")

                model.train()
                if rank == 0:
                    print(f"--- Validation finished. Resuming training... ---")

        if rank == 0:
            avg_epoch_loss = running_loss_epoch / steps_per_epoch
            print(f"Epoch {epoch+1} finish. Average training loss: {avg_epoch_loss:.4f}")

            log_entry = {'epoch': epoch + 1, 'avg_loss': avg_epoch_loss}
            epoch_results_log.append(log_entry)

            log_file_path = os.path.join(config.save_fold, "training_log.txt")
            try:
                with open(log_file_path, 'w') as f:
                    for entry in epoch_results_log:
                        f.write(str(entry) + '\n')
            except IOError as e:
                print(f"Warning: The logfile cannot be updated during epoch {epoch+1}. Reason: {e}")

    if rank == 0:
        print("\n--- Training Finished ---")
        print(f"Best Overall S-measure: {best_overall_sm:.4f} at Epoch: {best_epoch_for_overall}")


if __name__ == '__main__':
    main()
