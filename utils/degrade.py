import torch
import torch.nn as nn
import torch.nn.functional as FF
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import cv2
from PIL import Image
# from utils.rescaler import image_rescaler
from tqdm import tqdm

def jpeg_degrade(imgs: torch.Tensor, quality: int):
    """
    Degrade an image using JPEG compression.
    Parameters:
    - img: Input image tensor.
    Returns:
    - Degrade image tensor.
    
    Range of images are in [0, 1].
    """
    squeeze_flag = False
    device = imgs.device
    if len(imgs.shape) == 3: 
        imgs = imgs.unsqueeze(0)
        squeeze_flag = True
    imgs = imgs * 255 + .5
    imgs = imgs.clamp(0, 255).to(torch.uint8)
    
    results = []
    
    for img in imgs:
        img = img.permute(1, 2, 0).cpu().numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        _, stream = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results.append(torch.from_numpy(img).permute(2, 0, 1).to(torch.float32) / 255.)
    
    results = torch.stack(results)
    if squeeze_flag: results = results.squeeze(0).to(device)
    
    return results, len(stream) * 8 / (img.shape[0] * img.shape[1])
        
# img = torch.ones(16, 3, 256, 256).to(torch.float32) * 25 / 255
# print(jpeg_degrade(img, 50))
