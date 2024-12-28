

# Hyper parameters
num_gpu=1

vit_model=swin_base

dataset=imagenet
nb_classes=1000
patch_size=16
input_size=224
data_root=../data/imagenet


ft_batchsize=32
i=121
eval_resume="/home/xxu/adv_mae/ex_swin/swin_base"
checkp="/weight.pth"

m_port=`expr 12345 + $i`


CUDA_VISIBLE_DEVICES="4" python -m torch.distributed.launch --master_port $m_port --nproc_per_node=$num_gpu sub_imagenet_eval.py \
 --eval --resume "$eval_resume$checkp" \
 --model "$vit_model" \
 --batch_size $ft_batchsize \
 --data_root $data_root\
 --dataset "$dataset" --nb_classes $nb_classes \
 --patch_size $patch_size --input_size $input_size \
 --attack_train pgd --eps 4 --alpha 1 --aa_file "${eval_resume}/eval_aa_weight.txt"\
 --num_workers 10 > "${eval_resume}/eval_weight.txt" 2>&1



