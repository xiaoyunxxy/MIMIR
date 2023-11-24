# Hyper parameters

vit_model=vit_base

dataset=cifar10
nb_classes=10
patch_size=2
input_size=32
data_root=../data


ft_batchsize=16


eval_resume="./experiment/mae_cifar10_advfinetune_with_adv_fast_hsicpretrain"
checkp="/checkpoint-99.pth"


python finetune.py \
 --eval --resume "$eval_resume$checkp" \
 --model "$vit_model" \
 --batch_size $ft_batchsize \
 --data_root $data_root \
 --dataset "$dataset" --nb_classes $nb_classes \
 --patch_size $patch_size --input_size $input_size \
 --eps 8 --alpha 2 \
 --num_workers 4 > "${eval_resume}/eval.txt" 2>&1 

python finetune.py \
 --eval --resume "$eval_resume$checkp" \
 --model "$vit_model" \
 --batch_size $ft_batchsize \
 --data_root $data_root \
 --dataset "$dataset" --nb_classes $nb_classes \
 --patch_size $patch_size --input_size $input_size \
 --eps 8 --alpha 2 \
 --adap_eval \
 --num_workers 4 > "${eval_resume}/eval_adaptive.txt" 2>&1 