# export CUDA_VISIBLE_DEVICES=$1
# =================================================================================
# Train FADPNET
# =================================================================================
python train.py --gpus 2 --name fadpnet --model fadpnet \
    --Gnorm "bn" --lr 0.0002 --beta1 0.9 --scale_factor 8 --load_size 128 \
    --dataroot /path/to/datasets/CelebA_18k --dataset_name celeba --batch_size 32 --total_epochs 150 \
    --visual_freq 100 --print_freq 10 --save_latest_freq 500 #--continue_train 

# =================================================================================