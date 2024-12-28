import os
import torch
import numpy as np
import random

def distributed_init(args):
    '''This function performs the distributed setting.'''
    if args.distributed:
        if 'SLURM_PROCID' in os.environ:    # for distributed launch
            # print("os.environ['LOCAL_RANK']: ", os.environ['LOCAL_RANK'])
            # print("os.environ['SLURM_PROCID']: ", os.environ['SLURM_PROCID'])
            # print("os.environ['SLURM_NTASKS']: ", os.environ['SLURM_NTASKS'])
            # print("os.environ['RANK']: ", os.environ['RANK'])
            # print("os.environ['SLURM_NODELIST']: ", os.environ['SLURM_NODELIST'])
            # print("os.environ['WORLD_SIZE']: ", os.environ['WORLD_SIZE'])
            # print("os.environ['MASTER_PORT']: ", os.environ['MASTER_PORT'])
            # print("os.environ['MASTER_ADDR']: ", os.environ['MASTER_ADDR'])
            args.rank=int(os.environ['RANK'])
            args.device_id=int(os.environ['LOCAL_RANK'])
            args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        elif args.local_rank !=-1:    # for slurm scheduler
            args.rank=args.local_rank
            args.device_id=args.local_rank

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