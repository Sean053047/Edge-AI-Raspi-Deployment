import cv2
import numpy as np
import os.path as osp
from PIL import Image

import torch

from torch.utils.data import Dataset
from torchvision.transforms import Compose
from torch.nn import functional as F

from DepthAnythingV2.metric_depth.dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop
from DepthAnythingV2.metric_depth.util.metric import eval_depth
from DepthAnythingV2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2

from transformers import AutoImageProcessor, AutoModelForDepthEstimation

def init_model(encoder:str='vitb', max_depth=80.0):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }
    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_vkitti_{encoder}.pth', map_location='cpu'))
    return model

def init_huggingface_model(pretrained:str='depth-anything/Depth-Anything-V2-Small-hf'):
    image_processor = AutoImageProcessor.from_pretrained(pretrained)
    model = AutoModelForDepthEstimation.from_pretrained(pretrained)
    return image_processor, model

class CustomVKITTI2(Dataset):
    def __init__(self, data_root, filelist_path, mode, size=(518, 518)):
        self.mode = mode
        self.size = size
        self.data_root = data_root
        with open(filelist_path, 'r') as f:
            self.filelist = f.read().splitlines()
        
        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ] + ([Crop(size[0])] if self.mode == 'train' else []))
    
    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, item):
        img_path = osp.join(self.data_root, self.filelist[item].split(' ')[0])
        depth_path = osp.join(self.data_root, self.filelist[item].split(' ')[1])
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 100.0  # cm to m
        
        sample = self.transform({'image': image, 'depth': depth})

        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])
        sample['valid_mask'] = (sample['depth'] <= 80)
        
        sample['image_path'] = self.filelist[item].split(' ')[0]
        
        return sample
    
class CustomKITTI(Dataset):
    def __init__(self, data_root, filelist_path, mode, size=(518, 518)):
        self.mode = mode
        self.size = size
        self.data_root = data_root
        with open(filelist_path, 'r') as f:
            self.filelist = f.read().splitlines()
        
        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
    def __len__(self):
        return len(self.filelist)
    def __getitem__(self, item):
        img_path = osp.join(self.data_root, self.filelist[item].split(' ')[0])
        depth_path = osp.join(self.data_root, self.filelist[item].split(' ')[1])
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        depth = np.array(Image.open(depth_path), dtype=np.float32,) / 256
        sample = self.transform({'image': image, 'depth': depth})
        
        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])
        sample['depth'] = sample['depth']  # convert in meters
        
        sample['valid_mask'] = sample['depth'] > 0
        sample['image_path'] = self.filelist[item].split(' ')[0]
        
        return sample
    