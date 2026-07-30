# FADPNet: Frequency-Aware Dual-Path Network for Face Super-Resolution (TMM 2026)

## Installation and Requirements

We have trained and tested the codes on:
- Ubuntu 20.04
- CUDA 11.1  
- Python 3.8+

Install required packages:
```bash
pip install -r requirements.txt

## Note If you need to train from scratch, please install Mamba-ssm:
```bash
pip install causal-conv1d>=1.1.0
pip install mamba-ssm
```
## Train the Model

The commands used to train the released models are provided in script `train.sh`. Here are some training tips:

- You should download CelebA to train FADPNet. Please change the `--dataroot` to the path where your training images are stored.
- To train FADPNet, we simply crop out faces from CelebA without pre-alignment, because for ultra-low resolution face SR, it is difficult to pre-align the LR images.  
- Please change the `--name` option for different experiments. Tensorboard records with the same name will be moved to `check_points/log_archive`, and the weight directory will only store weight history of the latest experiment with the same name.
- If there's not enough memory, you can turn down the `--batch_size`.
- `--gpus` specifies the number of GPUs used for training. The script will use GPUs with more available memory first. To specify the GPU index, uncomment the `export CUDA_VISIBLE_DEVICES=`.

```bash
# Train FADPNet (8x scale)
CUDA_VISIBLE_DEVICES=0,1 python train.py --gpus 2 --name fadpnet --model fadpnet \
    --lr 0.0002 --beta1 0.9 --beta2 0.99 --scale_factor 8 --load_size 128 \
    --dataroot /path/to/datasets/CelebA --dataset_name celeba --batch_size 16 --total_epochs 150 \
    --visual_freq 100 --print_freq 10 --save_latest_freq 500
```
---

## Test the Models

```bash
# On CelebA Test set
python test.py --gpus 1 --model fadpnet --name fadpnet \
    --load_size 128 --dataset_name single --dataroot /path/to/datasets/test_datasets/CelebA1000/LR_x8_up/ \
    --pretrain_model_path ./pretrain_models/fadpnet/fadpnet_best.pth \
    --save_as_dir results_celeba/fadpnet
```

```bash
# On Helen Test set
python test.py --gpus 1 --model fadpnet --name fadpnet \
    --load_size 128 --dataset_name single --dataroot /path/to/datasets/test_datasets/Helen50/LR_x8_up/ \
    --pretrain_model_path ./pretrain_models/fadpnet/fadpnet_best.pth \
    --save_as_dir results_helen/fadpnet
```

### Evaluation

We provide evaluation codes in script `test.sh` to calculate PSNR/SSIM/LPIPS/VIF/Params/FLOPs scores.

## Acknowledgements

This code is built on [WFEN](https://github.com/IVIPLab/WFEN) and [MambaIR](https://github.com/csguoh/MambaIR). We thank the authors for sharing their codes.

---

## Citation

If you find this work useful for your research, please cite:

```bibtex
@article{xu2026fadpnet,
  title={FADPNet: Frequency-Aware Dual-Path Network for Face Super-Resolution},
  author={Xu, Siyu and Li, Wenjie and Gao, Guangwei and Yang, Jian and Qi, Guo-Jun and Lin, Chia-Wen},
  journal={IEEE Transactions on Multimedia},
  year={2026}
}
```
---

## :e-mail: Contact

If you have any questions, please email `xusiyu200107@163.com`.
