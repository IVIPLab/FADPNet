import os
import random
import numpy as np
from PIL import Image
import imgaug as ia
import imgaug.augmenters as iaa
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torchvision.transforms.functional as tf

from data.base_dataset import BaseDataset


class BlindSRDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        self.shuffle = True if opt.isTrain else False 
        self.lr_size = opt.load_size
        self.hr_size = opt.load_size

        self.lr_dir = opt.lr_dataroot    # 盲退化LR图像路径
        self.hr_dir = opt.hr_dataroot    # 对应HR图像路径
        
        # 获取图像列表（假设LR和HR文件名对应）
        self.img_names = self.get_img_names()

        # 数据增强（训练时）
        self.aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                Scale((1.0, 1.3), opt.load_size) 
                ])

        self.to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

    def get_img_names(self):
        img_names = [x for x in os.listdir(self.hr_dir) 
                     if x.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
        
        # 验证LR目录中是否存在对应文件
        valid_names = []
        for name in img_names:
            lr_path = os.path.join(self.lr_dir, name)
            if os.path.exists(lr_path):
                valid_names.append(name)
            else:
                # 尝试不同扩展名匹配
                base_name = os.path.splitext(name)[0]
                for ext in ['.png', '.jpg', '.jpeg']:
                    alt_path = os.path.join(self.lr_dir, base_name + ext)
                    if os.path.exists(alt_path):
                        valid_names.append(name)
                        break
        
        print(f"Found {len(valid_names)} paired images (HR+LR)")
        
        if self.shuffle:
            random.shuffle(valid_names)
        return valid_names

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        
        # 分别读取HR和LR（注意：LR已经是退化后的，不需要再下采样）
        hr_path = os.path.join(self.hr_dir, img_name)
        lr_path = os.path.join(self.lr_dir, img_name)
        #print(lr_path)
        
        # 如果扩展名不同，尝试查找
        if not os.path.exists(lr_path):
            base_name = os.path.splitext(img_name)[0]
            for ext in ['.png', '.jpg', '.jpeg']:
                alt_path = os.path.join(self.lr_dir, base_name + ext)
                if os.path.exists(alt_path):
                    lr_path = alt_path
                    break
        
        # 读取图像
        hr_img = Image.open(hr_path).convert('RGB')
        lr_img = Image.open(lr_path).convert('RGB')

        # 同步数据增强（确保LR和HR变换一致）
        if self.aug and self.opt.isTrain:
            seed = random.randint(0, 2**32)
            
            # 对HR进行增强
            random.seed(seed)
            torch.manual_seed(seed)
            hr_img = self.aug(hr_img)
            
            # 对LR进行相同的增强
            random.seed(seed)
            torch.manual_seed(seed)
            lr_img = self.aug(lr_img)

      
        hr_tensor = self.to_tensor(hr_img)
        lr_tensor = self.to_tensor(lr_img)

        return {
            'HR': hr_tensor, 
            'LR': lr_tensor, 
            'HR_paths': hr_path,
            'LR_paths': lr_path,
            'img_name': img_name}


# 保持Scale类用于可能的后续使用
class Scale(object):
    """
    Random scale the image and pad to the same size if needed.
    """
    def __init__(self, factor, size):
        self.factor = factor 
        rc_scale = (2 - factor[1], 1)
        self.size = (size, size)
        self.rc_scale = rc_scale
        self.ratio = (3. / 4., 4. / 3.) 
        self.resize_crop = transforms.RandomResizedCrop(size, rc_scale)

    def __call__(self, img):
        scale_factor = random.random() * (self.factor[1] - self.factor[0]) + self.factor[0]  
        w, h = img.size
        sw, sh = int(w*scale_factor), int(h*scale_factor)
        scaled_img = tf.resize(img, (sh, sw))
        if sw > w:
            i, j, h, w = self.resize_crop.get_params(img, self.rc_scale, self.ratio)
            scaled_img = tf.resized_crop(img, i, j, h, w, self.size, Image.BICUBIC) 
        elif sw < w:
            lp = (w - sw) // 2
            tp = (h - sh) // 2 
            padding = (lp, tp, w - sw - lp, h - sh - tp) 
            scaled_img = tf.pad(scaled_img, padding)
        return scaled_img