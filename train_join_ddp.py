import argparse
import os
import numpy as np
import random
from collections import OrderedDict
from tqdm import tqdm 
import torch.nn.functional as F
import torch
import torch.distributed as dist 
import torch.multiprocessing as mp 
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.cuda.amp import autocast, GradScaler
from models.SalMamba_tri import Model
from thop import profile
import IOU
from utils.init_func import group_weight
from get_dataloaders_ddp import setup_all_dataloaders_for_joint_training
from val_join_ddp import evaluate_all_tests_and_save_best 
from loss import structure_loss_edge,total_loss_suite_optimized
from collections import OrderedDict

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


def set_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 设置所有CUDA设备的种子
    np.random.seed(seed)
    random.seed(seed)

parser.add_argument(
    "--local_rank",
    type=int,
    default=int(os.environ.get("LOCAL_RANK", 0))
)

parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='mode')
config = parser.parse_args() 


def main():

    dist.init_process_group(backend='nccl')
    
    rank = dist.get_rank()      
    local_rank = config.local_rank 
    world_size = dist.get_world_size()

    if rank == 0:
        config.save_fold = config.save_fold + '/' + 'Joint_Training'
        if not os.path.exists("%s" % (config.save_fold)):
            os.makedirs("%s" % (config.save_fold), exist_ok=True) 


    fixed_model_init_seed = 1024 
    set_seed(fixed_model_init_seed) 
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
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
            print("Pre-trained model loaded successfully (with mismatches as detailed above).")

        except FileNotFoundError:
            print(f"Error: Pre-trained model file not found at {config.model_path}. Starting training from scratch.")
        except Exception as e:
            print(f"Error loading pre-trained model: {e}. Starting training from scratch.")
    
 
    dist.barrier() 
    if rank == 0:
        print(f"\n--- Rank {rank}: Model Parameters on {device} (after explicit move and SyncBatchNorm) before DDP ---")
    
    all_params_info = []
    for i, (name, param) in enumerate(model.named_parameters()):
        param_info = {
            'index': i,
            'name': name,
            'shape': tuple(param.shape),
            'stride': tuple(param.stride()),
            'device': param.device
        }
        all_params_info.append(param_info)

    if rank == 0:
        param_shapes_rank0 = {name: p.shape for name, p in model.named_parameters()}
        param_strides_rank0 = {name: p.stride() for name, p in model.named_parameters()}
    else:
        param_shapes_rank0 = None
        param_strides_rank0 = None

    gathered_shapes = [None for _ in range(world_size)]
    gathered_strides = [None for _ in range(world_size)]

    dist.gather_object(
        obj={name: p.shape for name, p in model.named_parameters()}, 
        object_gather_list=gathered_shapes if rank == 0 else None, 
        dst=0
    )
    dist.gather_object(
        obj={name: p.stride() for name, p in model.named_parameters()}, 
        object_gather_list=gathered_strides if rank == 0 else None, 
        dst=0
    )

    if rank == 0:
        print("\n--- Cross-Process Parameter Consistency Check (All Ranks - After Explicit Move & SyncBatchNorm) ---")
        for p_idx, (name, param) in enumerate(model.named_parameters()):
            current_shape = tuple(param.shape)
            current_stride = tuple(param.stride())
            
            is_shape_consistent = True
            is_stride_consistent = True

            for r in range(world_size):
                if gathered_shapes[r] is None or name not in gathered_shapes[r]:
                    print(f"WARNING: Param '{name}' (idx {p_idx}) not found on Rank {r} during gathering.")
                    is_shape_consistent = False
                    continue 

                if gathered_shapes[r][name] != current_shape:
                    print(f"ERROR: Param '{name}' (idx {p_idx}) shape mismatch across processes!")
                    print(f"  Rank 0 shape: {current_shape}")
                    print(f"  Rank {r} shape: {gathered_shapes[r][name]}")
                    is_shape_consistent = False
                
                if gathered_strides[r][name] != current_stride:
                    print(f"ERROR: Param '{name}' (idx {p_idx}) stride mismatch across processes!")
                    print(f"  Rank 0 stride: {current_stride}")
                    print(f"  Rank {r} stride: {gathered_strides[r][name]}")
                    is_stride_consistent = False
            
            if not is_shape_consistent or not is_stride_consistent:
                print(f"Full mismatch details for param '{name}' (idx {p_idx}):")
                for r in range(world_size):
                    shape_on_r = gathered_shapes[r].get(name, 'N/A') if gathered_shapes[r] else 'N/A'
                    stride_on_r = gathered_strides[r].get(name, 'N/A') if gathered_strides[r] else 'N/A'
                    print(f"  Rank {r}: Shape={shape_on_r}, Stride={stride_on_r}")
        print("--- End Consistency Check ---")
    
    dist.barrier() 

    model = model.to(device)
    for name, buffer in model.named_buffers():
        if buffer.device != device:
            buffer.data = buffer.data.to(device)
            print(f'Buffer {name} moved to {device} after SyncBatchNorm')
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True,broadcast_buffers=False) 

    dist.barrier()

    input1, input2, input3 ,input4,GT= torch.randn(1, 3, 448, 448), torch.randn(1, 3, 448, 448), torch.randn(1, 3, 448, 448), torch.zeros(1, 3, 448, 448), torch.randn(1, 1, 448, 448)
    input1, input2, input3, input4 ,GT= input1.to(device), input2.to(device), input3.to(device), input4.to(device), GT.to(device)
    mode = None
    input0 = None
    if rank == 0:
        flops, params = profile(model.module, inputs=(input1, input0, input1, input1, mode, GT))
        print(f"FLOPs: {flops}, Params: {params}")
    
    params_list = []
    params_list = group_weight(params_list, model.module, nn.BatchNorm2d, p['lr']) 

    optimizer = torch.optim.AdamW(params_list, lr=p['lr'], betas=(0.85, 0.999), weight_decay=p['wd'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = 10, 
            eta_min = 1e-5, 
            last_epoch = -1
        )


    scaler = GradScaler()
    optimizer.zero_grad()
    from torch.optim.lr_scheduler import ExponentialLR

    all_train_loaders, all_test_datasets = setup_all_dataloaders_for_joint_training(
        config, local_rank, world_size
    )

    train_iterators = {name: iter(dl) for name, dl in all_train_loaders.items()}

    if not all_train_loaders:
        if rank == 0:
            print("No training data loaders found. Exiting.")
        dist.destroy_process_group()
        return
    
    loader_lengths = [len(dl) for dl in all_train_loaders.values()] 
    num_tasks = len(all_train_loaders) # 找出最大批次数
    max_loader_len = max(loader_lengths) if loader_lengths else 0
    steps_per_epoch = num_tasks*max_loader_len
    accumulation_steps = 4
    
    if rank == 0:
        print("\n--- Training Configuration (New Logic) ---")
        print(f"Longest dataloader length: {max_loader_len} batches")
        print(f"Number of data modalities/tasks: {num_tasks}")
        print(f"Total steps per epoch is now: {max_loader_len} * {num_tasks} = {steps_per_epoch}")
        print("---------------------------------------------")


    task_names = sorted(list(all_train_loaders.keys()))
    epoch_results_log = []
    best_overall_sm = 0.0
    global_step = 0 
    validation_frequency = 1172 
    

    for epoch in range(config.epoch):
        model.train()
        running_loss_epoch = 0.0
        for loader in all_train_loaders.values():
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
                
                all_train_loaders[current_task_name].sampler.set_epoch(epoch)
                new_iterator = iter(all_train_loaders[current_task_name])
                train_iterators[current_task_name] = new_iterator
                data_batch = next(new_iterator)

            image = data_batch['image'].to(device, non_blocking=True)
            label = data_batch['label'].to(device, non_blocking=True)
  

            modal_flow_input = None
            modal_depth_input = None
            modal_thermal_input = None
            
            modal_depth_input = data_batch.get('depth')

            modal_flow_input = data_batch.get('flow')

            modal_thermal_input = data_batch.get('thermal')
            
            image, label = image.to(device), label.to(device)
            if modal_flow_input is not None:
                modal_flow_input = modal_flow_input.cuda(rank)

            if modal_depth_input is not None:
                modal_depth_input = modal_depth_input.cuda(rank)

            if modal_thermal_input is not None:   
                modal_thermal_input = modal_thermal_input.cuda(rank)
                          
            is_accumulation_step = ((step + 1) % accumulation_steps == 0)
            with autocast():
                out, saliency,aux_preds = model(image, modal_flow_input, modal_depth_input, modal_thermal_input, mode='train', gt=label)
            
                loss = total_loss_suite_optimized(out, label) + 0.7*structure_loss_edge(saliency,label)

                # aux_loss_total = 0.0
                # if aux_preds:
                #     for modal_name, pred_map in aux_preds.items():
                #         aux_loss_total += structure_loss_edge(pred_map, label)
                # loss += aux_loss_total*0.3
    
            
            if not is_accumulation_step:
                with model.no_sync():
                    scaler.scale(loss).backward()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            
            
            step_loss = loss.item()
            running_loss_epoch += step_loss
            if rank == 0:
           

                pbar.set_postfix(OrderedDict(
                    Loss=f"{step_loss:.4f}",
                    # MainLoss=f"{aux_loss_total:.4f}",
                    LR=f"{optimizer.param_groups[0]['lr']:.6f}",
                ))
            global_step += 1
                      
            if global_step > 0 and global_step % validation_frequency == 0:
                scheduler.step()
                if rank == 0:
                    print(f"\n--- Reached step {global_step}. Starting validation... ---")
                
                dist.barrier()             
                current_sm = evaluate_all_tests_and_save_best(model, all_test_datasets, epoch, config.save_fold, config)
                
                if rank == 0 and current_sm > best_overall_sm:
                    best_overall_sm = current_sm
                    best_epoch_for_overall = epoch + 1 
                    best_model_path = os.path.join(config.save_fold, f'best_model_step_0000.pth')
                    torch.save(model.module.state_dict(), best_model_path)
                    print(f"🎉 New best model saved to {best_model_path}, Step: {global_step}, S-measure: {best_overall_sm:.4f}")

                dist.barrier()
                model.train()
                if rank == 0:
                    print(f"--- Validation finished. Resuming training... ---")
        
        if steps_per_epoch % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

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

    dist.destroy_process_group()



    
if __name__ == '__main__':
    main()



