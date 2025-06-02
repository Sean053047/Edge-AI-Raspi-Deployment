import os# Finish Quantization script for DepthAnythingV2 model
import os.path as osp
import cv2
import numpy as np
from PIL import Image
from collections import defaultdict

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
# from transformers.modeling_outputs import DepthEstimatorOutput
# from transformers.models.dpt.image_processing_dpt import DPTImageProcessor

from DepthAnythingV2.metric_depth.dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop
from DepthAnythingV2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2

from utils import (
    init_model,
    set_seed,
    CustomKITTI,
)
from hqq_utils import AutoHQQTimmModel
from eval_model import evaluate
from torchinfo import summary 
import onnx

from hqq.core.quantize import BaseQuantizeConfig
def get_quant_config_deit(model, nbits=2, group_size=48):
    # * hqq supports quantization for torch.nn.Linear
    quant_config = {}
    n_blocks = len(model.pretrained.blocks)
    q2_config = BaseQuantizeConfig(nbits=nbits, group_size=group_size)
    for i in range(n_blocks):
        quant_config[f'pretrained.blocks.{i}.attn.qkv'] = q2_config
        quant_config[f'pretrained.blocks.{i}.attn.proj'] = q2_config
        quant_config[f'pretrained.blocks.{i}.mlp.fc1'] = q2_config
        quant_config[f'pretrained.blocks.{i}.mlp.fc2'] = q2_config
    return quant_config

def quantize_model(model, nbits=2, group_size=48, device='cuda'):
    model.device = device
    model.dtype = torch.float32 
    # * Quantize pretrained weights
    quant_config = get_quant_config_deit(model, nbits=nbits, group_size=group_size)
    AutoHQQTimmModel.quantize_model(model, quant_config=quant_config, compute_dtype=torch.float32, device=str(device))
    from hqq.utils.patching import prepare_for_inference
    prepare_for_inference(model)
    torch.cuda.empty_cache()
    return model

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Depth Estimation Model")
    parser.add_argument("--model", type=str, default='', help="Model name or path")
    parser.add_argument("--encoder", type=str, default="vitb", choices=["vits", "vitb", "vitl"], help="Encoder type for DepthAnythingV2")
    
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--nbits', type=int, default=4, help='Number of bits for quantization')
    parser.add_argument('--group-size', type=int, default=32, help='Group size for quantization')
    parser.add_argument('--max-depth', type=float, default=80.0, help='Maximum depth for normalization')
    parser.add_argument('--min-depth', type=float, default=0.001, help='Minimum depth for normalization')
    parser.add_argument("--grid-search", action='store_true', help="Enable grid search for quantization parameters")
    parser.add_argument("--quantize-only", action='store_true', help="Only quantize the model without evaluation")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use for evaluation")
    parser.add_argument("--save-dir", type=str, default='results', help="Directory to save results")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    filelist_path = 'kitti_val.txt'
    dataset = CustomKITTI(data_root='/data/kitti', filelist_path=filelist_path, mode='val', size=(518, 518))
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
    )
    set_seed(42)
    save_dir = osp.join(args.save_dir, args.encoder)
    os.makedirs(save_dir, exist_ok=True)
    if args.grid_search:
        import json
        nbits = [2, 4, 6]
        group_sizes = [32, 64, 96, 128]
        for bit, gsz in [(b, g) for b in nbits for g in group_sizes]:
            print(f"Testing with nbits={bit}, group_size={gsz}")
            model = init_model(encoder=args.encoder).to(device)
            model.eval()
            try:
                model = quantize_model(model, nbits=bit, group_size=gsz, device=device)
                results = evaluate(model, loader, device, min_depth=args.min_depth, max_depth=args.max_depth)
                with open(osp.join(save_dir, f'nbits_{bit}_group_size_{gsz}.json'), 'w') as f:
                    json.dump(results, f, indent=4)
            except Exception as e:
                print(f"Error with nbits={bit}, group_size={gsz}: {e}")
                continue
    else:
        model = init_model(encoder=args.encoder).to(device)
        model.eval()
        model = quantize_model(model, nbits=args.nbits, group_size=args.group_size, device=device)        
        if args.quantize_only:
            summary(model, input_size=(1, 3, 518, 1722), device=device)
        else:
            results = evaluate(model, loader, device, min_depth=args.min_depth, max_depth=args.max_depth)
            for k , v in results.items():
                print(f"{k}: {v:.4f}")
    # * (1, 3, 518, 1722) for KITTI
    
    
