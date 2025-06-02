import os.path as osp
import cv2
import numpy as np
from PIL import Image
from collections import defaultdict

import torch
from torch.nn import functional as F
from torch.utils.data import  DataLoader
# from transformers.modeling_outputs import DepthEstimatorOutput
# from transformers.models.dpt.image_processing_dpt import DPTImageProcessor

from DepthAnythingV2.metric_depth.dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop
from DepthAnythingV2.metric_depth.util.metric import eval_depth
from DepthAnythingV2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2

from utils import (
    init_model,
    init_huggingface_model,
    CustomKITTI,
    CustomVKITTI2
)

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