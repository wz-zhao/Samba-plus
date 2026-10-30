"""Run with python -m tests.ddp_smoke: exercise two real Gloo workers."""
import os
import socket
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.test_macl import TinyModel, config_for, make_data


def worker(rank, root, port):
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port),
                      RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE='2', USE_LIBUV='0')
    torch.set_num_threads(1)
    import train_macl_ddp as training
    import models.SalMamba_tri as model_module
    model_module.Model = TinyModel
    save = training.save_checkpoint

    def checked_save(model, *args, **kwargs):
        flat = torch.cat([p.detach().flatten() for p in model.parameters()])
        all_weights = [torch.empty_like(flat) for _ in range(2)]
        dist.all_gather(all_weights, flat)
        torch.testing.assert_close(all_weights[0], all_weights[1], atol=1e-7, rtol=1e-7)
        save(model, *args, **kwargs)

    training.save_checkpoint = checked_save
    try:
        training.run(config_for(Path(root)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='samba-ddp-') as directory:
        root = Path(directory)
        make_data(root)
        # An odd test size catches padded-duplicate and uneven-shard errors.
        path = root / 'rgb_dataset' / 'DUTS-TE' / 'test.txt'
        path.write_text(''.join(path.read_text().splitlines(keepends=True)[:5]))
        with socket.socket() as stream:
            stream.bind(('127.0.0.1', 0))
            port = stream.getsockname()[1]
        mp.spawn(worker, args=(directory, port), nprocs=2, join=True)
        checkpoint = torch.load(root / 'checkpoints/MACL_Training/last_checkpoint.pth',
                                map_location='cpu', weights_only=False)
        assert checkpoint['epoch'] == 3 and len(checkpoint['rank_states']) == 2
        print('Two-process DDP: stages, replay, remainder accumulation, uneven evaluation and synchronized weights passed')
