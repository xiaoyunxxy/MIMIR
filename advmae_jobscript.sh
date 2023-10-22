#!/bin/bash -e
#SBATCH --job-name preft_s_nor
#SBATCH --partition=icis
#SBATCH --account=icis
#SBATCH --qos=icis-preempt
#SBATCH --mem=200G
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH --time=7-12:00:00
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

dataset=imagenet
nb_classes=1000
patch_size=16
input_size=224
data_root=../data/imagenet

pre_batchsize=512
ft_batchsize=256

pre_blr=1.5e-4
ft_blr=0.001

pre_output_dir=./experiment_imagenet/${mae_model}_${dataset}_adv_fast_hsicpretrain
finetune_checkpoint=$pre_output_dir/checkpoint-799.pth
ft_output_dir=./experiment_imagenet/${mae_model}_${dataset}_advfinetune_with_adv_fast_hsicpretrain/

m_port=1234

# Pretrain

mkdir -p $pre_output_dir

OMP_NUM_THREADS=12 python -m torch.distributed.launch --master_port $m_port --nproc_per_node=$num_gpu pretrain.py \
--batch_size $pre_batchsize \
--model $mae_model \
--norm_pix_loss --mask_ratio 0.75 \
--epochs 800 --warmup_epochs 40 \
--blr $pre_blr --weight_decay 0.05 \
--dataset "$dataset" --patch_size $patch_size \
--data_root $data_root --input_size $input_size \
--output_dir "$pre_output_dir" --log_dir "$pre_output_dir" \
--attack pgd_mae --steps 1 --eps 4 --alpha 4 \
--num_workers 16 \
--use_normalize \
--mi_train hsic --mi_xpl 0.00001 > "$pre_output_dir/printlog" 2>&1


# Finetune

mkdir -p $ft_output_dir

OMP_NUM_THREADS=12 python -m torch.distributed.launch --master_port $m_port --nproc_per_node=$num_gpu finetune.py \
 --finetune "$finetune_checkpoint" \
 --model "$vit_model" \
 --output_dir "$ft_output_dir" \
 --log_dir "$ft_output_dir" \
 --batch_size $ft_batchsize \
 --epochs 100 --warmup_epochs 10 \
 --blr $ft_blr \
 --layer_decay 0.65 \
 --weight_decay 0.05 --drop_path 0.1 \
 --reprob 0.0 --aa 'noaug' \
 --data_root $data_root \
 --dataset "$dataset" --nb_classes $nb_classes \
 --patch_size $patch_size --input_size $input_size \
 --attack_train pgd --eps 4 --alpha 4 --steps 1 \
 --use_normalize \
 --num_workers 16 > "$ft_output_dir/printlog"

