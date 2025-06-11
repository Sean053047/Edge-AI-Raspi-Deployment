# TODO: Finish the FPS testing script for .onnx files
from utils import set_seed, init_model
from hqq_quantize import get_quant_config_deit, quantize_model
import cv2
import numpy as np

def depth2image(depth, max_depth, min_depth):
    depth = (depth - min_depth) / (max_depth - min_depth) * 255.0
    return depth.astype(np.uint8)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Depth Estimation Model")
    parser.add_argument("option", type=str, choices=['raw', 'quantized', 'dump'], help="Option to run the model: 'raw' for unquantized, 'quantized' for quantized model")
    parser.add_argument("video", type=str,  help="Path to the video file for FPS testing")
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
    if args.option == 'quantized':
        quant_config = get_quant_config_deit(model, nbits=args.nbits, group_size=args.group_size)
        model = quantize_model(model, quant_config=quant_config, device=device)        
    
    import time 
    time_record = []
    
    cap = cv2.VideoCapture(args.video)
    # img_shape = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # dump_video = cv2.VideoWriter('dump.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, img_shape)
    print(f"FPS testing start ...")
    frame_count = 0
    while frame_count < 10:  
        st_time = time.time()
        ret, frame = cap.read()
        frame_count += 1
        if not ret: break
        depth = model.infer_image(frame, input_size=518)
        time_record.append(time.time() - st_time)
        print(f"Frame {frame_count}: Time taken: {time_record[-1]:.4f} seconds")
        # depth_im = depth2image(depth, args.max_depth, args.min_depth)
        # cv2.imshow('Depth Map', depth_im)
        
    cap.release()
    print('FPS:', frame_count / sum(time_record))
    
    import matplotlib.pyplot  as plt
    plt.plot(time_record)
    plt.xlabel('Frame')
    plt.ylabel('Time (s)')
    plt.title('Time per Frame')
    plt.savefig(f'{args.option}_time_per_frame.png')
    # depth_im = depth2image(depth, args.max_depth, args.min_depth)
    # cv2.imwrite('depth.jpg', depth_im)
    