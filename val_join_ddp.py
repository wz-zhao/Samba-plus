"""Single-process/DDP evaluation without padded duplicate test samples."""
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from Smeasure import s_measure

best_overall_sm = -1.0
best_epoch_for_overall = -1


def evaluate_all_tests_and_save_best(model, all_test_datasets, epoch, save_path, config):
    global best_overall_sm, best_epoch_for_overall
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    raw_model = model.module if hasattr(model, 'module') else model
    device = next(raw_model.parameters()).device
    was_training = model.training
    model.eval()
    scores = {}
    try:
        with torch.no_grad():
            for name, dataset in all_test_datasets.items():
                # Unwrap DDP: uneven evaluation shards must not run forward collectives.
                shard = Subset(dataset, range(rank, len(dataset), world_size))
                loader = DataLoader(shard, batch_size=getattr(config, 'test_batch_size', 2),
                                    num_workers=config.num_thread, pin_memory=device.type == 'cuda')
                totals = torch.zeros(2, device=device, dtype=torch.float64)
                for batch in loader:
                    image = batch['image'].to(device, non_blocking=True)
                    label = batch['label'].to(device, non_blocking=True)
                    inputs = [batch[key].to(device, non_blocking=True) if key in batch else None
                              for key in ('flow', 'depth', 'thermal')]
                    output, _, _ = raw_model(image, *inputs, mode='val', task=name)
                    if isinstance(output, (tuple, list)):
                        output = output[0]
                    for prediction, target in zip(output.sigmoid(), label):
                        minimum, maximum = prediction.min(), prediction.max()
                        prediction = (prediction - minimum) / (maximum - minimum + 1e-8)
                        totals[0] += s_measure(prediction, target).double()
                        totals[1] += 1
                if distributed:
                    dist.all_reduce(totals)
                if int(totals[1].item()) != len(dataset):
                    raise RuntimeError(f'Evaluation sample count mismatch for {name}')
                if len(dataset) == 0:
                    raise ValueError(f'Empty evaluation dataset: {name}')
                scores[name] = (totals[0] / totals[1]).item()
                if rank == 0:
                    print(f'{name} S-measure: {scores[name]:.4f}')
        average = sum(scores.values()) / len(scores) if scores else 0.0
        if rank == 0 and scores:
            destination = Path(save_path)
            destination.mkdir(parents=True, exist_ok=True)
            with (destination / 'validation.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(epoch=epoch + 1, stage=getattr(config, '_validation_stage', None),
                                             overall_avg_sm=average, **scores)) + '\n')
            if average > getattr(config, '_best_validation_sm', -1.0):
                best_overall_sm, best_epoch_for_overall = average, epoch + 1
                config._best_validation_sm = average
                torch.save(raw_model.state_dict(), destination / 'overall_best.pth')
        return average
    finally:
        model.train(was_training)
