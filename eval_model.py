import os.path as osp
import cv2
import numpy as np
from PIL import Image
from collections import defaultdict

import torch
from torch.nn import functional as F
from torchvision.transforms import Compose
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
# from transformers.modeling_outputs import DepthEstimatorOutput
# from transformers.models.dpt.image_processing_dpt import DPTImageProcessor

from DepthAnythingV2.metric_depth.dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop
from DepthAnythingV2.metric_depth.util.metric import eval_depth
from DepthAnythingV2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2

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

# * Valid depth range for outdoor scene: [0.01, 80.0]
@ torch.no_grad()
def evaluate(model, loader, device="cuda", min_depth=0.001, max_depth=80):
    '''batch_size=1 is necessary for evaluation. Some images are in different sizes.'''
    from tqdm import tqdm
    
    num_sample = 0
    results = {'d1': 0.0, 'd2': 0.0, 'd3': 0.0, 
                'abs_rel': 0.0, 'sq_rel': 0.0, 'rmse': 0.0, 
                'rmse_log': 0.0, 'log10': 0.0, 'silog': 0.0}
    
    for batch in tqdm(loader, desc="Validation:", total=len(loader)):
        image = batch['image'].to(device).float()
        depth = batch['depth'].to(device).float()
        valid_mask = batch['valid_mask'].to(device).float()
        valid_mask = (valid_mask == 1) & (depth >= min_depth) & (depth <= max_depth)
        if valid_mask.sum() < 10: continue
        
        height, width = depth.shape[1], depth.shape[2]
        pred_depth = model(image)
        pred_depth = F.interpolate(pred_depth.unsqueeze(1), size=(height, width), mode='bilinear', align_corners=False).squeeze(1)
        
        metric = eval_depth(pred_depth[valid_mask], depth[valid_mask])
        num_sample += 1
        for key in results.keys():
            results[key] += metric[key]
    results = {k: v / num_sample for k, v in results.items()}
    return results

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Depth Estimation Model")
    parser.add_argument("--model", type=str, default='', help="Model name or path")
    parser.add_argument("--encoder", type=str, default="vitb", choices=["vits", "vitb", "vitl"], help="Encoder type for DepthAnythingV2")
    parser.add_argument("--kitti-root", type=str, default=None, help="Root directory for KITTI dataset")
    parser.add_argument("--vkitti2-root", type=str, default=None, help="Root directory for VKITTI2 dataset")
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--max-depth', type=float, default=80.0, help='Maximum depth for normalization')
    parser.add_argument('--min-depth', type=float, default=0.001, help='Minimum depth for normalization')
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use for evaluation")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    model = init_model(encoder=args.encoder).to(device)
    # * Initialize dataloader
    if args.kitti_root is not None:
        filelist_path = 'kitti_val.txt'
        dataset = CustomKITTI(data_root=args.kitti_root, filelist_path=filelist_path, mode='val', size=(518, 518))
    elif args.vkitti2_root is not None:
        filelist_path = 'vkitti2_train.txt'
        dataset = CustomVKITTI2(data_root=args.vkitti2_root, filelist_path=filelist_path, mode='val', size=(518, 518))
    else:
        raise ValueError("Please specify either --kitti-root or --vkitti2-root")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
    )
    results = evaluate(model, loader, device, min_depth=args.min_depth, max_depth=args.max_depth)
    print("Evaluation Results:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")