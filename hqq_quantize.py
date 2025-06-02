# Finish Quantization script for DepthAnythingV2 model
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
def get_quant_config_deit(model):
    # * hqq supports quantization for torch.nn.Linear
    quant_config = {}
    n_blocks = len(model.pretrained.blocks)
    q2_config = BaseQuantizeConfig(nbits=2, group_size=48)
    for i in range(n_blocks):
        quant_config[f'pretrained.blocks.{i}.attn.qkv'] = q2_config
        quant_config[f'pretrained.blocks.{i}.attn.proj'] = q2_config
        quant_config[f'pretrained.blocks.{i}.mlp.fc1'] = q2_config
        quant_config[f'pretrained.blocks.{i}.mlp.fc2'] = q2_config
    return quant_config

def main(model, loader, device, min_depth=0.001, max_depth=80.0):
    set_seed(42)
    model.device = device
    model.dtype = torch.float32 
    
    # * Quantize pretrained weights
    quant_config = get_quant_config_deit(model)
    AutoHQQTimmModel.quantize_model(model, quant_config=quant_config, compute_dtype=torch.float32, device=str(device))

    from hqq.utils.patching import prepare_for_inference
    prepare_for_inference(model)
    torch.cuda.empty_cache()
    
    results = evaluate(model, loader, device, min_depth=min_depth, max_depth=max_depth)
    for k , v in results.items():
        print(f"{k}: {v:.4f}")

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Depth Estimation Model")
    parser.add_argument("--model", type=str, default='', help="Model name or path")
    parser.add_argument("--encoder", type=str, default="vitb", choices=["vits", "vitb", "vitl"], help="Encoder type for DepthAnythingV2")
    
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--max-depth', type=float, default=80.0, help='Maximum depth for normalization')
    parser.add_argument('--min-depth', type=float, default=0.001, help='Minimum depth for normalization')
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use for evaluation")
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
    model = init_model(encoder=args.encoder).to(device)
    model.eval()
    main(model, loader, device, max_depth=args.max_depth, min_depth=args.min_depth)
    # * (1, 3, 518, 1722) for KITTI
    # summary(model, input_size=(1, 3, 518, 1722), device=device.type)
    
