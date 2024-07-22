import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from dataset.dataset import *
from metrics import *
from models.net import Falcon as net



# Parser
def set_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="None", type=str)
    parser.add_argument('--sh_file', default="None", type=str)
    parser.add_argument('--result_dir', default="None", type=str)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--wandb', default=0, type=int)
    parser.add_argument('--ddp', default=0, type=int)
    args = parser.parse_args()
    assert args.config_path!=None, "Config Path is not assigned"
    return args

def main(args, config):
    test_hazy_img_list = sortedFileList(Path.cwd()/config['dataset']['name']/config['dataset']['test']['hazy'])
    test_gt_img_list = sortedFileList(Path.cwd()/config['dataset']['name']/config['dataset']['test']['gt'])
    test_dataset = Img_Dataset(hazy_img_list=test_hazy_img_list, 
                            hazy_root_dir=Path.cwd()/config['dataset']['name']/config['dataset']['test']['hazy'], 
                            gt_img_list=test_gt_img_list, 
                            gt_root_dir=Path.cwd()/config['dataset']['name']/config['dataset']['test']['gt'], 
                            transform=ImageTransformForTest(),
                            mask=config['train']['mask'])
    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle=False)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print("Device : ", device)

    ffc = config.pop('ffc')
    model = net(4, 3, config_ffc=ffc)


    which={"best" : Path.cwd()/'ckpt'/config['env']['name']/'dense.pth',
        }

    Path(Path(config["result_dir"])/config["env"]["name"]).mkdir(exist_ok=True)
    for k, path in which.items():
        print(f">> testing when {k} <<")
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model"])
        model = model.to(device)
        model.eval()
        count = 0
        total_psnr = 0
        total_ssim = 0
        with torch.no_grad():
            for _, pair in enumerate(test_dataloader):
                count += 1
                if(config['train']['mask']):
                    hazy = pair["hazy_img"]
                else:
                    hazy = pair["hazy_img"]
                gt = pair["gt_img"]
                hazy = hazy.to(device, non_blocking=True).float()
                gt = gt.to(device, non_blocking=True).float()
                
                start_time = time.time()
                dehazed = model(hazy, test=True)
                end_time = time.time()
                
                gt = gt.cpu().numpy().squeeze()
                dehazed = dehazed.cpu().numpy().squeeze()
                
                gt = gt.transpose((1, 2, 0))
                dehazed = dehazed.transpose((1, 2, 0))
                
                gt = gt * 0.5 + 0.5
                dehazed = dehazed * 0.5 + 0.5
                dehazed = np.clip(dehazed, 0, 1)
                
                _psnr = peak_signal_noise_ratio(gt, dehazed, data_range = 1.0)
                total_psnr += _psnr
                
                _ssim = structural_similarity(gt, dehazed, data_range = 1.0, channel_axis = 2)
                total_ssim += _ssim
                
                print(f"img{_} : PSNR>{_psnr} SSIM>{_ssim}")
        print(f"PSNR> {total_psnr / count}")
        print(f"SSIM> {total_ssim / count}")

if __name__ == "__main__":
    args = set_parser()
    print(args)
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    main(args, config)