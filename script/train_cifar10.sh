# Hyper parameters
num_gpu=4
mae_model=mae_vit_small
vit_model=vit_small

dataset=cifar10
nb_classes=10
patch_size=2
input_size=32
data_root=../data

pre_batchsize=128
ft_batchsize=32

pre_blr=1.5e-4
ft_blr=0.05

pre_output_dir=./experiment/${mae_model}_${dataset}_adv_fast_hsicpretrain_edm
finetune_checkpoint=$pre_output_dir/checkpoint-799.pth
ft_output_dir=./experiment/${mae_model}_${dataset}_advedmfinetune_with_adv_fast_hsicpretrain_edm

edm_data="./edm_data/cifar10/5m.npz"

m_port=1234

# Pretrain

mkdir -p $pre_output_dir


python -m torch.distributed.launch --master_port $m_port --nproc_per_node=$num_gpu pretrain.py \
--batch_size $pre_batchsize \
--model $mae_model \
--norm_pix_loss --mask_ratio 0.75 \
--epochs 800 --warmup_epochs 40 \
--blr $pre_blr --weight_decay 0.05 \
--dataset "$dataset" --patch_size $patch_size \
--data_root $data_root --input_size $input_size \
--output_dir "$pre_output_dir" --log_dir "$pre_output_dir" \
--attack pgd_mae --steps 1 --eps 8 --alpha 10 \
--num_workers 0 \
--use_edm --aux_data_filename $edm_data \
--mi_train hsic --mi_xpl 0.00001 > "${pre_output_dir}/printlog"

# Finetune

mkdir -p $ft_output_dir

python -m torch.distributed.launch --master_port $m_port --nproc_per_node=$num_gpu finetune.py \
 --finetune "$finetune_checkpoint" \
 --model "$vit_model" \
 --output_dir "$ft_output_dir" \
 --log_dir "$ft_output_dir" \
 --batch_size $ft_batchsize \
 --epochs 50 --warmup_epochs 5 \
 --blr $ft_blr \
 --layer_decay 0.65 \
 --weight_decay 0.05 --drop_path 0.1 \
 --reprob 0.0 --aa 'noaug'\
 --data_root $data_root \
 --dataset "$dataset" --nb_classes $nb_classes \
 --patch_size $patch_size --input_size $input_size \
 --attack_train pgd --eps 8 --alpha 2 --steps 10 \
 --use_edm --aux_data_filename $edm_data \
 --num_workers 0 > "${ft_output_dir}/printlog"


