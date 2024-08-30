import os
import torch
import torch.distributed as dist
from typing import List, Optional

def _get_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))

def _get_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))

def maybe_init_dist() -> Optional[int]:
    try:
        # provided by torchrun
        rank = _get_rank()
        world_size = _get_world_size()

        if world_size < 2:
            # too few gpus to parallelize, tp is no-op
            return None
    except KeyError:
        # not run via torchrun, no-op
        return None

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank

def run(rank, size):
    """ Distributed function to be implemented. """
    # Set the GPU device for this process
    torch.cuda.set_device(rank)

    # Create a tensor on the GPU
    tensor = torch.zeros(1).cuda(rank)
    if rank == 0:
        tensor += 1
    dist.broadcast(tensor, src=0)
    print(f'Rank {rank} has data {tensor[0].item()}')

def main():
    rank = maybe_init_dist()
    size = _get_world_size()
    run(rank, size)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
