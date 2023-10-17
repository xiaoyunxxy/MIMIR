#!/bin/bash -e
#SBATCH --job-name pre_ft
#SBATCH --partition=icis
#SBATCH --account=icis
#SBATCH --qos=icis-preempt
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --time=5-12:00:00
#SBATCH --output=./slurm_log/my-experiment-%j.out
#SBATCH --error=./slurm_log/my-experiment-%j.err
#SBATCH --mail-user=xiaoyun.xu@ru.nl
#SBATCH --mail-type=BEGIN,END,FAIL

source /vol/tensusers2/xxu/pytorch/bin/activate
cd /home/xxu/adv_mae


# Hyper parameters
num_gpu=4
mae_model=mae_vit_ti
vit_model=vit_ti

dataset=imagenet
nb_classes=1000
patch_size=16
input_size=224
data_root=../data/imagenet

pre_batchsize=512
ft_batchsize=256

pre_blr=1.5e-4
ft_blr=0.005

pre_output_dir=./experiment/mae_imagenet_adv_fast_hsicpretrain1
finetune_checkpoint=$pre_output_dir/checkpoint-799.pth
ft_output_dir=./experiment/${mae_model}_${dataset}_advfinetune_with_adv_fast_hsicpretrain_fix1/


# Pretrain

# mkdir -p $pre_output_dir

# OMP_NUM_THREADS=12 python -m torch.distributed.launch --nproc_per_node=$num_gpu pretrain.py \
# --batch_size $pre_batchsize \
# --model $mae_model \
# --norm_pix_loss --mask_ratio 0.75 \
# --epochs 800 --warmup_epochs 40 \
# --blr $pre_blr --weight_decay 0.05 \
# --dataset "$dataset" --patch_size $patch_size \
# --data_root $data_root --input_size $input_size \
# --output_dir "$pre_output_dir" --log_dir "$pre_output_dir" \
# --attack pgd_mae --steps 1 --alpha 0.0392 \
# --num_workers 16 \
# --mi_train hsic --mi_xpl 0.00001 > "$pre_output_dir/printlog" 2>&1


# Finetune

echo 'fix'

mkdir -p $ft_output_dir

OMP_NUM_THREADS=6 python -m torch.distributed.launch --nproc_per_node=$num_gpu finetune.py \
 --finetune "$finetune_checkpoint" \
 --model "$vit_model" \
 --output_dir "$ft_output_dir" \
 --log_dir "$ft_output_dir" \
 --batch_size $ft_batchsize \
 --epochs 50 \
 --blr $ft_blr \
 --layer_decay 0.65 \
 --weight_decay 0.05 --drop_path 0.1 \
 --reprob 0.25 \
 --data_root $data_root \
 --dataset "$dataset" --nb_classes $nb_classes \
 --patch_size $patch_size --input_size $input_size \
 --attack_train trades --eps 0.0157 --alpha 0.0157 --steps 1 \
 --num_workers 10 > "$ft_output_dir/printlog" 2>&1




#  ---------------------
# pre_output_dir=./experiment/mae_imagenet_adv_fast_hsicpretrain_small_epsalpha
# finetune_checkpoint=$pre_output_dir/checkpoint-799.pth
# ft_output_dir=./experiment/${mae_model}_${dataset}_advfinetune_with_adv_fast_hsicpretrain_small_epsalpha/

# Pretrain

# echo 'pre'

# mkdir -p $pre_output_dir

# OMP_NUM_THREADS=12 python -m torch.distributed.launch --nproc_per_node=$num_gpu pretrain.py \
# --batch_size $pre_batchsize \
# --model $mae_model \
# --norm_pix_loss --mask_ratio 0.75 \
# --epochs 800 --warmup_epochs 40 \
# --blr $pre_blr --weight_decay 0.05 \
# --dataset "$dataset" --patch_size $patch_size \
# --data_root $data_root --input_size $input_size \
# --output_dir "$pre_output_dir" --log_dir "$pre_output_dir" \
# --attack pgd_mae --eps 0.0157 --alpha 0.0078 --steps 1 \
# --num_workers 10 \
# --mi_train hsic --mi_xpl 0.00001 > "$pre_output_dir/printlog" 2>&1


# # Finetune

# echo 'ft'

# mkdir -p $ft_output_dir

# OMP_NUM_THREADS=12 python -m torch.distributed.launch --nproc_per_node=$num_gpu finetune.py \
#  --finetune "$finetune_checkpoint" \
#  --model "$vit_model" \
#  --output_dir "$ft_output_dir" \
#  --log_dir "$ft_output_dir" \
#  --batch_size $ft_batchsize \
#  --epochs 50 \
#  --blr $ft_blr \
#  --layer_decay 0.65 \
#  --weight_decay 0.05 --drop_path 0.1 \
#  --reprob 0.25 \
#  --data_root $data_root \
#  --dataset "$dataset" --nb_classes $nb_classes \
#  --patch_size $patch_size --input_size $input_size \
#  --attack_train pgd --eps 0.0157 --alpha 0.0078 --steps 1 \
#  --num_workers 10 > "$ft_output_dir/printlog" 2>&1

