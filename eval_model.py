from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from transformers.modeling_outputs import DepthEstimatorOutput
from transformers.models.dpt.image_processing_dpt import DPTImageProcessor
import torch
import numpy as np
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")

# prepare image for the model
inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
# interpolate to original size and visualize the prediction
post_processed_output = image_processor.post_process_depth_estimation(
    outputs,
    target_sizes=[(image.height, image.width)],
)

predicted_depth = post_processed_output[0]["predicted_depth"]
depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
depth = depth.detach().cpu().numpy() * 255
depth = Image.fromarray(depth.astype("uint8"))
depth.save("depth_anything_output.png")



def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Depth Estimation Model")
    parser.add_argument("--model", type=str, default="depth-anything/Depth-Anything-V2-Small-hf", help="Model name or path")
    return parser.parse_args()

if __name__ == "__main__":
    ...