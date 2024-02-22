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
                
        hr_patch = self.totensor(image.copy())
        
        lr_image: torch.Tensor = image_rescaler(image.copy(), 255., 255., self.interpolation, 1 / self.scale_factor, self.antialias, 
                                                output_format='array', dtype='uint8', layout='hwc', dispersed=True)
        
        lr_patch = self.totensor(lr_image.copy())
        
        
        ehr_patch, ehr_bitrate = self.get_jpeg(image)
        elr_patch, elr_bitrate = self.get_jpeg(lr_image)
        
        if self.cache:
            self.cache[idx] = (hr_patch, lr_patch, ehr_patch, elr_patch, ehr_bitrate, elr_bitrate)
        
        return hr_patch, lr_patch, ehr_patch, elr_patch, ehr_bitrate, elr_bitrate
    
    def get_jpeg(self, img):
        _, bitstream = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, self.img_quality])
        decoded = cv2.imdecode(bitstream, cv2.IMREAD_COLOR)
        decoded = self.totensor(decoded)
        return decoded, len(bitstream) * 8 / (img.shape[0] * img.shape[1])
    
    def totensor(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img).to(torch.float32).permute(2, 0, 1) / 255.
                    


        

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
    