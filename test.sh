# export CUDA_VISIBLE_DEVICES=$1
# ================================================================================
# Test FADPNET on Helen and CelebA test dataset
# ================================================================================

python test.py --gpus 1 --model fadpnet --name fadpnet \
    --load_size 128 --dataset_name single --dataroot /path/to/datasets/test_datasets/Helen50/LR_x8_up/ \
    --pretrain_model_path ./check_points/fadpnet.pth \
    --save_as_dir results_helen/fadpnet

python test.py --gpus 1 --model fadpnet --name fadpnet \
    --load_size 128 --dataset_name single --dataroot /path/to/datasets/test_datasets/CelebA1000/LR_x8_up/ \
    --pretrain_model_path ./check_points/fadpnet.pth \
    --save_as_dir results_celeba/fadpnet

# ----------------- calculate PSNR/SSIM scores ----------------------------------
python psnr_ssim.py
# ------------------------------------------------------------------------------- 

# ----------------- calculate LPIPS/VIF scores ----------------------------------
python vif_lpips/lpips_2dirs.py --use_gpu
python vif_lpips/VIF.py
# ------------------------------------------------------------------------------- 

# ----------------- calculate Parmas/FLOPS scores -------------------------------
python calc_flops.py
# ------------------------------------------------------------------------------- 
