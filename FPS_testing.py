# TODO: Finish the FPS testing script for .onnx files
from utils import set_seed, init_model
from hqq_quantize import get_quant_config_deit, quantize_model

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
    
    parser.add_argument("--device", type=str, choices=['gpu', 'cpu'], help="Device to use for evaluation (e.g., 'cuda', 'cpu')")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use for evaluation")
    parser.add_argument("--save-dir", type=str, default='results', help="Directory to save results")
    return parser.parse_args()    

if __name__ == "__main__":
    args = parse_args()
    device = 'cpu'
    set_seed(42)
    
    model = init_model(encoder=args.encoder).to(device)
    model.eval()
    quant_config = get_quant_config_deit(model, nbits=args.nbits, group_size=args.group_size)
    model = quantize_model(model, quant_config=quant_config, device=device)        
    
    