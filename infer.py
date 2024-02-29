import torch
import torch
import torch.nn as nn
import torch.nn.functional as FF
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import os.path as osp
import time
import argparse
parser = argparse.ArgumentParser()
import warnings
import os
import yaml
warnings.filterwarnings("ignore")
import importlib
from utils.dataset import ImageDataset as ds
from utils.degrade import jpeg_degrade
from utils.rescaler import image_rescaler
torch.backends.cudnn.benchmark = True

parser.add_argument('--cuda', type=int, default=3, help='CUDA device index')
parser.add_argument('-q', type=int, default=80, help='quality')
parser.add_argument('--workers', type=int, default=1, help='number of workers for dataloader')
args = parser.parse_args()
device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
quality = int(args.q)
workers = int(args.workers)

with open('./config/infer.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
    

load_model_dir = config['general']['load_model_dir']
log_dir = config['general']['log_dir']

os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=f'{log_dir}/{quality}')


def load_model(module_prefix, config, **extra_args):
    module_name = f"{module_prefix}.{config['name']}"
    module = importlib.import_module(module_name)
    model_class = getattr(module, config['name'])
    return model_class(**config.get('args', {}), **extra_args)


crnet = load_model('models.crnet', config['crnet'], scale=config['general']['scale'])
ppnet = load_model('models.ppnet', config['ppnet'], scale=config['general']['scale'])
acnet = load_model('models.acnet', config['acnet'], num_levels=config['acnet']['num_levels'])
benet = load_model('models.benet', config['benet'], m=config['benet']['m'])





crnet.to(device); ppnet.to(device); acnet.to(device); benet.to(device)

criterion = nn.MSELoss()

models = [crnet, ppnet, acnet, benet]
model_type = ['crnet', 'ppnet', 'acnet', 'benet']


train_config = {
    'crnet': {
        'train': False,
        'model': crnet,
        'get_input': lambda hr, lr, ehr, elr, bih, bil: (hr, lr), 
    },
    'ppnet': {
        'train': False,
        'model': ppnet,
        'get_input': lambda hr, lr, ehr, elr, bih, bil: (elr, hr), 
    },
    'acnet': {
        'train': False,
        'model': acnet,
        'get_input': lambda hr, lr, ehr, elr, bih, bil: (hr, ehr), 
    },
    'benet': {
        'train': False,
        'model': benet,
        'get_input': lambda hr, lr, ehr, elr, bih, bil: (hr, bih), 
    }
}




def validate(epoch, quality):
    val_dataset = ds(config['general']['val_dirs'], 0, True, False, config['general']['interpolation'],
                    config['general']['scale_factor'], config['general']['antialias'], quality)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    psnr: torch.Tensor = lambda x: torch.log10(1 / torch.tensor(x)) * 10. if x > 0 else 0
    for model in models: model.eval()
    psnr_rec, psnr_deg, bitrate = 0, 0, 0
    with torch.no_grad():
        for i, (hr, lr, eh, el, bih, bil) in enumerate(val_loader):
            hr, lr, eh, el, bih, bil = hr.to(device), lr.to(device), eh.to(device), el.to(device), \
                                        bih.to(device, dtype=torch.float32), bil.to(device, dtype=torch.float32)
            cr_result = crnet(hr)
            ac_result = acnet(cr_result)
            degraded_imgs, degraded_bitrate = jpeg_degrade(cr_result, quality)
            if config['general']['post_only']: degraded_imgs, degraded_bitrate = el, bil.item()
            pp_result = ppnet(degraded_imgs.to(device))
                        
            psnr_rec += psnr(criterion(pp_result, hr))
            
            psnr_deg += psnr(criterion(degraded_imgs.cpu(), ac_result.cpu()))
            bitrate += degraded_bitrate            
        
        met = [0, psnr_rec, psnr_deg, bitrate]
        for _ in range(4):
            met[_] /= len(val_loader) 
            if met[_] != 0: 
                writer.add_scalar(f'val/psnr_{model_type[_]}', met[_], epoch)
        if met[0] != 0: writer.add_images('imgs/cr_images', cr_result.cpu(), epoch)
        if met[1] != 0: writer.add_images('imgs/pp_images', pp_result.cpu(), epoch)
        if met[2] != 0: 
            writer.add_images('imgs/ac_images', ac_result.cpu(), epoch)
        for _ in range(3): met[_] = met[_].item() if isinstance(met[_], torch.Tensor) else met[_]
        print(f'quality {quality}: Epoch {epoch} | PSNR: CR {met[0]:.2f} | PP {met[1]:.2f} | AC {met[2]:.2f} | BE {met[3]:.2f}')




best_results = [0, 0, 0, -1]
iter = 0

def execute():
    global quality
    for q in range(10, 90, 10):
        quality = q
        for i, model in enumerate(models):
            if (not config['general']['post_only']) or model_type[i]=='ppnet': model.load_state_dict(torch.load(osp.join(load_model_dir, str(quality), model_type[i], model.__class__.__name__ + '.pth')))
        current_results = validate(1, q)


    

execute()