from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from transformers.modeling_outputs import DepthEstimatorOutput
from transformers.models.dpt.image_processing_dpt import DPTImageProcessor
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from torch.nn import functional as F

import os.path as osp
import cv2
import numpy as np
from PIL import Image
import requests
from collections import defaultdict
from DepthAnythingV2.metric_depth.dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop
from DepthAnythingV2.metric_depth.util.metric import eval_depth
from DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2

from datasets import load_dataset

class CustomVIKITTI2(Dataset):
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
                keep_aspect_ratio=False,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ] + ([Crop(size[0])] if self.mode == 'train' else []))
    def __len__(self):
        return len(self.filelist)
    def __getitem__(self, idx):
        img_path = osp.join(self.data_root, self.filelist[idx].split(' ')[0])
        depth_path = osp.join(self.data_root, self.filelist[idx].split(' ')[1])
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 100.0  # cm to m
        
        sample = self.transform({'image': image, 'depth': depth})

        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])
        
        sample['valid_mask'] = (sample['depth'] <= 80)
        
        sample['image_path'] = self.filelist[idx].split(' ')[0]
        return sample

def init_model(encoder:str='vitb'):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    return model

def init_huggingface_model(pretrained:str='depth-anything/Depth-Anything-V2-Small-hf'):
    image_processor = AutoImageProcessor.from_pretrained(pretrained)
    model = AutoModelForDepthEstimation.from_pretrained(pretrained)
    return image_processor, model


@ torch.no_grad()
def evaluate(args):
    batch_size = 1
    filelist_path = 'vkitti2_train.txt'
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    dataset = CustomVIKITTI2(data_root=args.vkitti2_root, filelist_path=filelist_path, mode='test', size=(518, 518))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    depth_metrics = defaultdict(list)    
    
    model = init_model(encoder=args.encoder).to(device)
    for batch in dataloader:
        image = batch['image'].to(device)
        depth = batch['depth'].to(device)
        valid_mask = batch['valid_mask'].to(device)
        height, width = depth.shape[1], depth.shape[2]
        
        pred_depth = model(image)
        pred_depth = F.interpolate(pred_depth.unsqueeze(1), size=(height, width), mode='bicubic', align_corners=False).squeeze(1)
        
        pred_depth = pred_depth[valid_mask] 
        depth = depth [valid_mask]
        print(pred_depth.shape, depth.shape)
        print(pred_depth.shape, depth.shape)
        metric = eval_depth(pred_depth, depth)
        print(metric)
        exit()
            
    

official_model = "depth-anything/Depth-Anything-V2-Small-hf"

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Depth Estimation Model")
    parser.add_argument("--pretrained", type=str, default="depth-anything/Depth-Anything-V2-Small-hf", help="Model name or path")
    parser.add_argument("--vkitti2_root", type=str, default="/data//vkitti2", help="Root directory for VKITTI2 dataset")
    parser.add_argument("--encoder", type=str, default="vitb", choices=["vits", "vitb", "vitl"], help="Encoder type for DepthAnythingV2")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use for evaluation")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
    print("Depth estimation completed and saved as 'depth_anything_output.png'.")