import os
import torch
import numpy as np
import random

def distributed_init(args):
    '''This function performs the distributed setting.'''
    if args.distributed:
        if args.local_rank !=-1:    # for distributed launch
            print('args.local_rank !=-1. ')
            args.rank = int(os.environ["RANK"])
            args.world_size = int(os.environ['WORLD_SIZE'])
            args.device_id = int(os.environ['LOCAL_RANK'])
        elif 'SLURM_PROCID' in os.environ:
            # print('SLURM_PROCID: ', os.environ['SLURM_PROCID'])
            args.rank=int(os.environ['SLURM_PROCID'])
            args.device_id=args.rank % torch.cuda.device_count()

        torch.cuda.set_device(args.device_id)
        args.dist_backend = 'nccl'
        print('| distributed init (rank {}): {}, gpu {}'.format(
            args.rank, args.dist_url, args.local_rank), flush=True)
        torch.distributed.init_process_group(backend=args.dist_backend)
        torch.distributed.barrier()
        setup_for_distributed(args.rank==0)
    else:
        args.local_rank=0
        args.world_size=1
        args.rank=0
        args.device_id=0
        torch.cuda.set_device(args.device_id)
        
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print
    
def random_seed(seed=0, rank=0):
    seed = seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)