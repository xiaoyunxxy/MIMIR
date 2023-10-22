#!/bin/bash -e
#SBATCH --job-name preft_c_edm
#SBATCH --partition=icis
#SBATCH --account=icis
#SBATCH --qos=icis-preempt
#SBATCH --mem=150G
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=./slurm_log/my-experiment-%j.out
#SBATCH --error=./slurm_log/my-experiment-%j.err
#SBATCH --mail-user=xiaoyun.xu@ru.nl
#SBATCH --mail-type=BEGIN,END,FAIL

source /ceph/dis-ceph/xxu/pytorch/bin/activate
cd /home/xxu/adv_mae


# Hyper parameters
num_gpu=4
mae_model=mae_vit_small
vit_model=vit_small

dataset=cifar10
nb_classes=10
patch_size=2
input_size=32
data_root=../data

pre_batchsize=256
ft_batchsize=32

pre_blr=1.5e-4
ft_blr=0.05

pre_output_dir=./experiment/${mae_model}_${dataset}_adv_fast_hsicpretrain_edm
finetune_checkpoint=$pre_output_dir/checkpoint-799.pth
ft_output_dir=./experiment/${mae_model}_${dataset}_advedmfinetune_with_adv_fast_hsicpretrain_edm

m_port=1239

# Pretrain

mkdir -p $pre_output_dir

OMP_NUM_THREADS=2 python -m torch.distributed.launch --master_port $m_port --nproc_per_node=$num_gpu pretrain.py \
--batch_size $pre_batchsize \
--model $mae_model \
--norm_pix_loss --mask_ratio 0.75 \
--epochs 800 --warmup_epochs 40 \
--blr $pre_blr --weight_decay 0.05 \
--dataset "$dataset" --patch_size $patch_size \
--data_root $data_root --input_size $input_size \
--output_dir "$pre_output_dir" --log_dir "$pre_output_dir" \
--attack pgd_mae --steps 1 --eps 10 --alpha 8 \
--num_workers 16 \
--use_edm --aux_data_filename "/ceph/dis-ceph/xxu/edm_data/cifar10/5m.npz" \
--mi_train hsic --mi_xpl 0.00001 > "${pre_output_dir}/printlog" 2>&1


# Finetune

mkdir -p $ft_output_dir

OMP_NUM_THREADS=2 python -m torch.distributed.launch --master_port $m_port --nproc_per_node=$num_gpu finetune.py \
 --finetune "$finetune_checkpoint" \
 --model "$vit_model" \
 --output_dir "$ft_output_dir" \
 --log_dir "$ft_output_dir" \
 --batch_size $ft_batchsize \
 --epochs 100 \
 --blr $ft_blr \
 --layer_decay 0.65 \
 --weight_decay 0.05 --drop_path 0.1 \
 --reprob 0.25 --use_edm \
 --aux_data_filename "/ceph/dis-ceph/xxu/edm_data/cifar10/5m.npz" \
 --data_root $data_root \
 --dataset "$dataset" --nb_classes $nb_classes \
 --patch_size $patch_size --input_size $input_size \
 --attack_train pgd \
 --num_workers 16 > "${ft_output_dir}/printlog" 2>&1
