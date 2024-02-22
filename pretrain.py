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
torch.backends.cudnn.benchmark = True

parser.add_argument('--cuda', type=int, default=3, help='CUDA device index')
parser.add_argument('-q', type=int, default=90, help='quality')
parser.add_argument('--workers', type=int, default=1, help='number of workers for dataloader')
args = parser.parse_args()
device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
quality = int(args.q)
workers = int(args.workers)

with open('./config/pretraining.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
    
model_dir = config['general']['model_dir']
log_dir = config['general']['log_dir']

os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=f'logs/{quality}')


def load_model(module_prefix, config, **extra_args):
    module_name = f"{module_prefix}.{config['name']}"
    module = importlib.import_module(module_name)
    model_class = getattr(module, config['name'])
    return model_class(**config.get('args', {}), **extra_args)

# 假设config是一个配置字典，包含所有模型的配置信息
crnet = load_model('models.crnet', config['crnet'], scale=config['general']['scale'])
ppnet = load_model('models.ppnet', config['ppnet'], scale=config['general']['scale'])
acnet = load_model('models.acnet', config['acnet'], num_levels=config['acnet']['num_levels'])
benet = load_model('models.benet', config['benet'], m=config['benet']['m'])

train_crnet = config['crnet']['train']
train_ppnet = config['ppnet']['train']
train_acnet = config['acnet']['train']
train_benet = config['benet']['train']

train_dataset = ds(config['general']['train_dirs'], config['general']['crop_size'], False, 
                   True, config['general']['interpolation'], config['general']['scale_factor'],
                   config['general']['antialias'], quality)
val_dataset = ds(config['general']['val_dirs'], 0, True, False, config['general']['interpolation'],
                    config['general']['scale_factor'], config['general']['antialias'], quality)
test_dataset = ds(config['general']['test_dirs'], 0, True, False, config['general']['interpolation'],
                    config['general']['scale_factor'], config['general']['antialias'], quality)


train_loader = DataLoader(train_dataset, batch_size=config['general']['batch_size'], shuffle=True, num_workers=workers, 
                          pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0,pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=workers,pin_memory=True)

crnet.to(device); ppnet.to(device); acnet.to(device); benet.to(device)

criterion = nn.MSELoss()

models = [crnet, ppnet, acnet, benet]
model_type = ['crnet', 'ppnet', 'acnet', 'benet']
optimizers = [torch.optim.Adam(net.parameters(), lr=config['general']['lr'], betas=(0.9, 0.99)) \
              for net, mt in zip(models, model_type) if config[mt]['train']]
for net in models:
    net.train()

train_config = {
    'crnet': {
        'train': train_crnet,
        'model': crnet,
        'optimizer': torch.optim.Adam(crnet.parameters(), lr=config['general']['lr'], betas=(0.9, 0.99)),
        'get_input': lambda hr, lr, ehr, elr, bih, bil: (hr, lr), 
    },
    'ppnet': {
        'train': train_ppnet,
        'model': ppnet,
        'optimizer': torch.optim.Adam(ppnet.parameters(), lr=config['general']['lr'], betas=(0.9, 0.99)),
        'get_input': lambda hr, lr, ehr, elr, bih, bil: (elr, hr), 
    },
    'acnet': {
        'train': train_acnet,
        'model': acnet,
        'optimizer': torch.optim.Adam(acnet.parameters(), lr=config['general']['lr'], betas=(0.9, 0.99)),
        'get_input': lambda hr, lr, ehr, elr, bih, bil: (hr, ehr), 
    },
    'benet': {
        'train': train_benet,
        'model': benet,
        'optimizer': torch.optim.Adam(benet.parameters(), lr=config['general']['lr'], betas=(0.9, 0.99)),
        'get_input': lambda hr, lr, ehr, elr, bih, bil: (hr, bih), 
    }
}

def iter_process(typ, data, add_log=False, val=False):
    if not train_config[typ]['train']: return 0., 0.
    x, y = train_config[typ]['get_input'](*data)
    yhat = train_config[typ]['model'](x)
    loss = criterion(yhat, y)
    if not val:
        loss.backward()
        train_config[typ]['optimizer'].step()
        train_config[typ]['optimizer'].zero_grad()
        if add_log:
            writer.add_scalar(f'train/loss_{typ}', loss.item(), iter)
    else:
        quanted_yhat = (yhat * 255 + .5).clamp(0, 255).to(torch.uint8).to(torch.float32) / 255.
        quanted_loss = criterion(quanted_yhat, y)
        results = loss.item(), quanted_loss.item()
        return results
    

def validate(epoch):
    psnr: torch.Tensor = lambda x: torch.log10(1 / torch.tensor(x)) * 10. if x > 0 else 0
    for model in models: model.eval()
    with torch.no_grad():
        met = [0, 0, 0, 0]; met_quanted = [0, 0, 0, 0]
        for i, (hr, lr, eh, el, bih, bil) in enumerate(val_loader):
            hr, lr, eh, el, bih, bil = hr.to(device), lr.to(device), eh.to(device), el.to(device), \
                                        bih.to(device, dtype=torch.float32), bil.to(device, dtype=torch.float32)
            l_cr = iter_process('crnet', (hr, lr, eh, el, bih, bil), False, True); met[0] += psnr(l_cr[0]); met_quanted[0] += psnr(l_cr[1])
            l_pp = iter_process('ppnet', (hr, lr, eh, el, bih, bil), False, True); met[1] += psnr(l_pp[0]); met_quanted[1] += psnr(l_pp[1])
            l_ac = iter_process('acnet', (hr, lr, eh, el, bih, bil), False, True); met[2] += psnr(l_ac[0]); met_quanted[2] += psnr(l_ac[1])
            l_be = iter_process('benet', (hr, lr, eh, el, bih, bil), False, True); met[3] -= l_be[0]
        for _ in range(4):
            met[_] /= len(val_loader) 
            if met[_] != 0: 
                writer.add_scalar(f'val/psnr_{model_type[_]}', met[_], epoch)
        if met[0] != 0: writer.add_images('imgs/cr_images', crnet(hr).cpu(), epoch)
        if met[1] != 0: writer.add_images('imgs/pp_images', ppnet(el).cpu(), epoch)
        if met[2] != 0: 
            writer.add_images('imgs/ac_images', acnet(hr).cpu(), epoch)
            writer.add_scalar('val/psnr_acnet_quanted', met_quanted[2] / len(val_loader), epoch)
        for _ in range(3): met[_] = met[_].item() if isinstance(met[_], torch.Tensor) else met[_]
        print(f'quality {quality}: Epoch {epoch} | PSNR: CR {met[0]:.2f} | PP {met[1]:.2f} | AC {met[2]:.2f} | BE {met[3]:.2f}')
    for model in models: model.train()
    return met

best_results = [0, 0, 0, -1]
iter = 0

def train(epoch):
    global iter
    for model in models: model.train()
    # for i, (hr, lr, eh, el, bih, bil) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)):
    for i, (hr, lr, eh, el, bih, bil) in enumerate(train_loader):
        iter = iter + 1
        hr, lr, eh, el, bih, bil = hr.to(device), lr.to(device), eh.to(device), el.to(device), \
                                        bih.to(device, dtype=torch.float32), bil.to(device, dtype=torch.float32)
        make_log = iter % config['general']['summary_iters'] == 0
        iter_process('crnet', (hr, lr, eh, el, bih, bil), make_log)
        iter_process('ppnet', (hr, lr, eh, el, bih, bil), make_log)
        iter_process('acnet', (hr, lr, eh, el, bih, bil), make_log)
        iter_process('benet', (hr, lr, eh, el, bih, bil), make_log)
    current_results = validate(epoch)
    for i, (result, model, typ) in enumerate(zip(current_results, models, model_type)):
        if result > best_results[i] and config[typ]['train'] and config[typ]['save']:
            best_results[i] = result
            os.makedirs(osp.join(model_dir, str(quality), typ), exist_ok=True)
            torch.save(model.state_dict(), osp.join(model_dir, str(quality), typ, f'{model.__class__.__name__}.pth'))
            print(f'quality {quality}: Best {typ} model saved, epoch {epoch}, file name: {quality}/{typ}/{model.__class__.__name__}.pth')

for epoch in range(config['general']['epochs']):
    train(epoch)