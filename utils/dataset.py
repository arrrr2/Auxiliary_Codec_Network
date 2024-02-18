import torch
import torch.nn as nn
import torch.nn.functional as FF
from torch.utils.data import Dataset, DataLoader
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import cv2
from PIL import Image
from utils.rescaler import image_rescaler
from tqdm import tqdm


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dirs, crop_size=256, if_cache=False, augument=True,
                 interpolation='bilinear', scale_factor=1, antialias=False, 
                 quality=100):

        self.root_dirs = root_dirs
        self.if_cache = if_cache
        self.crop_size = crop_size
        self.augument = augument
        self.interpolation = interpolation
        self.scale_factor = scale_factor
        self.antialias = antialias
        self.images = []
        self.cache = {}
        self.img_quality = quality
        for root_dir in root_dirs:
            for root, _, files in os.walk(root_dir):
                for file in files:
                    file: str
                    if file.lower().endswith(('.png','.bmp','.jpg')):
                        self.images.append(os.path.join(root, file))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        if self.if_cache:
            if idx in self.cache:
                return self.cache[idx]

        image_path = self.images[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image_height, image_width, _ = image.shape
        crop_height, crop_width = self.crop_size, self.crop_size
        top = np.random.randint(0, image_height - crop_height + 1)
        left = np.random.randint(0, image_width - crop_width + 1)
        if self.crop_size != 0: image = image[top:top+crop_height, left:left+crop_width]
        else:
            crop_height = image_height - image_height % self.scale_factor
            crop_width = image_width - image_width % self.scale_factor
            crop_height = crop_height - crop_height % 16
            crop_width = crop_width - crop_width % 16
            image = image[:crop_height, :crop_width]
        
        if self.augument:
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=0)
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=1)
            if np.random.rand() > 0.5:
                image = np.rot90(image)
                
        hr_patch = torch.from_numpy(image.copy()).to(torch.float32).permute(2, 0, 1).contiguous() / 255.
        
        lr_patch: torch.Tensor = image_rescaler(hr_patch, 1, 1, self.interpolation, 1 / self.scale_factor, self.antialias)
        
        elr_patch = lr_patch.clone() * 255 + 0.5
        elr_patch = elr_patch.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous().numpy()
        elr_patch = cv2.cvtColor(elr_patch, cv2.COLOR_RGB2BGR)

        # Encode the LR patch into a bitstream
        _, bitstream = cv2.imencode('.jpg', elr_patch, [cv2.IMWRITE_JPEG_QUALITY, self.img_quality])
        # Decode the bitstream back into an image
        decoded = cv2.imdecode(bitstream, cv2.IMREAD_COLOR)
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        decoded = torch.from_numpy(decoded).to(torch.float32).permute(2, 0, 1).contiguous() / 255.
        elr_patch = decoded
        
        bitrate = len(bitstream) * 8 / (crop_height * crop_width)
        
        if self.cache:
            self.cache[idx] = (hr_patch, lr_patch, elr_patch, bitrate)
        
        return hr_patch, lr_patch, elr_patch, bitrate
                    


        

if __name__=='__main__':
    dataset = ImageDataset(['/mnt/datasets/coco/unlabeled2017'], crop_size=224, if_cache=False, augument=True, scale_factor=2, quality=1)
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=64)
    for hr, lr, elr, bitrate in tqdm(dataloader):
        pass
        # print(hr.shape, lr.shape, elr.shape, bitrate)
        # # save the hr, lr, elr to png files, with torchvision
        # torchvision.utils.save_image(hr, 'hr.png', normalize=True)
        # torchvision.utils.save_image(lr, 'lr.png', normalize=True)
        # torchvision.utils.save_image(elr, 'elr.png', normalize=True)
        # break
    