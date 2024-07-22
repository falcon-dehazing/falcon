import argparse
import gc

import os
import os.path as osp
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from dataset.dataset import *
from losses.perceptual import PerceptualNet
from models.net import Falcon as net
from scheduler.scheds import CosineAnnealingWarmUpRestarts
from utils import calculate_remaining_time, set_seed, setup_logging

def set_parser():
    parser = argparse.ArgumentParser()  
    parser.add_argument('--config_path', default="None", type=str)
    parser.add_argument('--sh_file', default="None", type=str)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--wandb', default=0, type=int)
    parser.add_argument('--ddp', default=0, type=int)
    parser.add_argument('--port', default=24660, type=int)
    args = parser.parse_args()
    assert args.config_path!=None, "Config Path is not assigned"
    return args

def get_optimizer(config, model, lr):
    optimizers = {
        "adam": optim.Adam(model.parameters(), lr=lr, weight_decay=float(config['opt']['wd'])),
        "adamw": optim.AdamW(model.parameters(), lr=lr, weight_decay=float(config['opt']['wd'])),
        "sgd" : optim.SGD(model.parameters(), lr=lr, weight_decay=float(config['opt']['wd'])),
        # can extend more optimizers
    }
    return optimizers.get(config['opt']['name'].lower(), None)

def get_scheduler(scheduler_name, optimizer, config):
    schedulers = {
        "cawr2" : CosineAnnealingWarmUpRestarts(optimizer, 
                                            T_0=int(config['train']['epoch']*config['sched']['cawr2']['t0']), 
                                            T_mult=config['sched']['cawr2']['tmul'], 
                                            eta_max=config['sched']['lr_max'], 
                                            T_up=int(config['train']['epoch']*config['sched']['cawr2']['tup']), 
                                            gamma=config['sched']['cawr2']['gamma']),
        "cawr" :  torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                                    T_0=int(config['train']['epoch']/config['sched']['cawr']['t0']), 
                                                                    T_mult=config['sched']['cawr']['tmul'], 
                                                                    eta_min=config['sched']['lr_max'], 
                                                                    last_epoch=-1),
        "steplr" : torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                    milestones=config['sched']['steplr_multi']['milestones'], 
                                                    gamma=config['sched']['steplr_multi']['gamma'])
    }
    return schedulers.get(scheduler_name, None)

def main(args, config):
    print(args)
    set_seed(config["seed"])
    local_gpu_id = torch.cuda.current_device()
    
    # DataLoader Setting
    in_size = tuple(config['dataset']['in_size'])
    train_hazy_img_list = sortedFileList(Path.cwd()/config['dataset']['name']/config['dataset']['train']['hazy'])
    train_gt_img_list = sortedFileList(Path.cwd()/config['dataset']['name']/config['dataset']['train']['gt'])
    
    val_hazy_img_list = sortedFileList(Path.cwd()/config['dataset']['val']['name']/config['dataset']['val']['hazy'])
    val_gt_img_list = sortedFileList(Path.cwd()/config['dataset']['val']['name']/config['dataset']['val']['gt'])

    test_hazy_img_list = sortedFileList(Path.cwd()/config['dataset']['test']['name']/config['dataset']['test']['hazy'])
    test_gt_img_list = sortedFileList(Path.cwd()/config['dataset']['test']['name']/config['dataset']['test']['gt'])

    train_dataset = Img_Dataset(hazy_img_list=train_hazy_img_list, 
                            hazy_root_dir=Path.cwd()/config['dataset']['name']/config['dataset']['train']['hazy'], 
                            gt_img_list=train_gt_img_list, 
                            gt_root_dir=Path.cwd()/config['dataset']['name']/config['dataset']['train']['gt'], 
                            transform=ImageTransform(in_size),
                            mask=config['train']['mask'])
    val_dataset = Img_Dataset(hazy_img_list=val_hazy_img_list, 
                            hazy_root_dir=Path.cwd()/config['dataset']['val']['name']/config['dataset']['val']['hazy'], 
                            gt_img_list=val_gt_img_list, 
                            gt_root_dir=Path.cwd()/config['dataset']['val']['name']/config['dataset']['val']['gt'], 
                            transform=ImageTransform(in_size),
                            mask=config['train']['mask'])
    test_dataset = Img_Dataset(hazy_img_list=test_hazy_img_list, 
                            hazy_root_dir=Path.cwd()/config['dataset']['test']['name']/config['dataset']['test']['hazy'], 
                            gt_img_list=test_gt_img_list, 
                            gt_root_dir=Path.cwd()/config['dataset']['test']['name']/config['dataset']['test']['gt'], 
                            transform=ImageTransformForTest(),
                            mask=config['train']['mask'],
                            for_test=True)

    train_dataloader = DataLoader(train_dataset, batch_size = config['train']['batch'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['train']['batch'], shuffle=False)
    
    # Device Setting
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_gpu_id}')
    else:
        device = torch.device('cpu')

    print("Device : ", device)

    # Model & Criterion & Optimizer & Scheduler Setting
    ffc = config.pop('ffc')
    model = net(len(config['train']['input_kernel'])+3, 3, input_kernel=config['train']['input_kernel'], config_ffc=ffc).cuda(local_gpu_id)
    mse = nn.MSELoss()
    
    optimizer = get_optimizer(config, model, float(config['sched']['init_lr']) )
    scheduler = get_scheduler(config['sched']['name'].lower(), optimizer, config)

    # Training Loop
    num_epochs = config['train']['epoch']
    perc_loss_network = PerceptualNet(net=config['train']['perceptual']['net'], style_layers=config['train']['perceptual']['style'], content_layers=config['train']['perceptual']['content'], device=device)
    os.makedirs(Path.cwd()/config['result_dir']/config['env']['name'], exist_ok=True)
    
    # Train
    start_ep = 0
    loss_ratio = tuple(map(float, config['train']['loss_ratio']))
    for epoch in range(start_ep, num_epochs):
        start_time=time.time()
        model.train()
        print(f"\nrank:{local_gpu_id} Epoch [{epoch}/{num_epochs-1}] train start! @{config['env']['name']} @{args.sh_file} @{args.config_path}")
        for _, pair in enumerate(train_dataloader):
            hazy = pair["hazy_img"].to(device, non_blocking=True).float()
            gt = pair["gt_img"].to(device, non_blocking=True).float()
            t_gt = pair["t_gt"].to(device, non_blocking=True).float()

            dehazed, t_haze = model(hazy, w=config['train']['w'])
            
            loss_img = mse(dehazed, gt) # MSELoss
            loss_map = mse(t_haze.to(device), t_gt.to(device))
            loss_perc = perc_loss_network(dehazed, gt) if config['train']['perceptual']['net'] is not False else torch.Tensor([0.]).to(device)
            loss_final = loss_ratio[0] * loss_img + loss_ratio[1] * loss_map + loss_ratio[2] * loss_perc
            
            optimizer.zero_grad()
            loss_final.backward()
            optimizer.step()
        scheduler.step()
        # Validation
        if local_gpu_id == 0 and epoch%5==0 :
            model.eval()
            print(f"Epoch [{epoch}/{num_epochs-1}] validation start!")
            
            with torch.no_grad():
                total_loss = 0.0
                num_samples = 0
                
                for _, pair in enumerate(val_dataloader):
                    hazy = pair["hazy_img"]
                    gt = pair["gt_img"]
                    t_gt = pair['t_gt']
                    
                    hazy = hazy.to(device, non_blocking=True).float()
                    gt = pair["gt_img"]
                    hazy = hazy.to(device, non_blocking=True).float()
                    gt = gt.to(device, non_blocking=True).float()
                    t_gt = pair["t_gt"]
                    dehazed, t_haze = model(hazy)
                    loss_img = mse(dehazed, gt) # MSE
                    loss_map = mse(t_haze.to(device), t_gt.to(device)) # MSE
                    loss_perc = perc_loss_network(dehazed, gt) if config['train']['perceptual']['net'] is not False else torch.Tensor([0.]).to(device)                    
                    loss_final = loss_ratio[0] * loss_img + loss_ratio[1] * loss_map + loss_ratio[2] * loss_perc
                    num_samples += hazy.size(0)
                    total_loss += loss_final.item() * hazy.size(0)
            
            avg_loss = total_loss / num_samples
            
            print(f"Epoch [{epoch}/{num_epochs-1}], Avg. Loss: {avg_loss:.4f}")
            
            # Model Save
            save_dict = {
                    "model" : model.state_dict(),
                    "opt":optimizer.state_dict(),
                    "sched":scheduler.state_dict(),
                    "ep" : epoch,
                }
            torch.save(save_dict, Path.cwd()/config['result_dir']/config['env']['name']/'latest.pth')
            
        gc.collect()
        torch.cuda.empty_cache()
        calculate_remaining_time(start_time, epoch, num_epochs)


if __name__ == "__main__":
    args = set_parser()
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    main(args, config)