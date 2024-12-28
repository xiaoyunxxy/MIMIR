#!/bin/bash -e


echo "----"
echo ${MASTER_PORT}
echo ${MASTER_ADDR}
echo ${SLURM_NODEID}
echo ${SLURM_JOB_NUM_NODES}
echo "----"


OMP_NUM_THREADS=1 python -m torch.distributed.launch \
	--master_port=${MASTER_PORT} \
	--nproc_per_node=4 \
	--nnodes=${SLURM_JOB_NUM_NODES} \
	--node_rank=${SLURM_NODEID} \
	--master_addr=${MASTER_ADDR} \
	adversarial_training.py --configs=./train_configs/swin_large_patch4_window7_224.yaml
