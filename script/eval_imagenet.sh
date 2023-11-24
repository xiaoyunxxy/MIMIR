# Hyper parameters

vit_model=vit_small

dataset=imagenet
nb_classes=1000
patch_size=16
input_size=224
data_root=../data/imagenet


ft_batchsize=16


eval_resume="./experiment_imagenet/mae_vit_small_imagenet_advfinetune_with_adv_fast_hsicpretrain"
checkp="/checkpoint-99.pth"


python finetune.py \
 --eval --resume "$eval_resume$checkp" \
 --model "$vit_model" \
 --batch_size $ft_batchsize \
 --data_root $data_root \
 --dataset "$dataset" --nb_classes $nb_classes \
 --patch_size $patch_size --input_size $input_size \
 --attack_train pgd --eps 2 --alpha 1 \
 --num_workers 4 > "${eval_resume}/eval.txt" 2>&1 
