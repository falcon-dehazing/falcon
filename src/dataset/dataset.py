import os
from os import path
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

import dcp

class ImageTransform():
    def __init__(self, in_size):
        self.data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    def __call__(self, img, norm=True):
        if norm == False:
            transforms_excluding_normalize = transforms.Compose([
                                                t for t in self.data_transform.transforms
                                                if not isinstance(t, (transforms.Normalize))
                                            ])
            return transforms_excluding_normalize(img)
        return self.data_transform(img)

class ImageTransformForTest():
    def __init__(self):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    def __call__(self, img, norm=True):
        if norm == False:
            transforms_excluding_normalize = transforms.Compose([
                                                t for t in self.data_transform.transforms
                                                if not isinstance(t, (transforms.Normalize))
                                            ])
            return transforms_excluding_normalize(img)
        return self.data_transform(img)
    
class Img_Dataset(data.Dataset):
    def __init__(self, hazy_img_list, hazy_root_dir, gt_img_list, gt_root_dir, transform, mask, for_test=False):
        self.hazy_img_list = hazy_img_list
        self.hazy_root_dir = hazy_root_dir
        self.gt_img_list = gt_img_list
        self.gt_root_dir = gt_root_dir
        self.transform = transform
        self.mask = mask
        self.for_test = for_test
        if ('ITS' in str(self.hazy_root_dir)) or ('SOTS' in str(self.hazy_root_dir)):
            self.gt_img_dict = {int(name.split('.')[0]) : name for idx, name in enumerate(gt_img_list)}
    def colorJitter(self, img):
        img = img*0.5 + 0.5 
        return transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(transforms.ColorJitter(brightness=.2, hue=.2, saturation=.2)(img))
    def __len__(self):
        if (len(self.hazy_img_list) == len(self.gt_img_list)):
            return len(self.hazy_img_list)
        else:
            return len(self.hazy_img_list)
    def __getitem__(self, index):
        if ('ITS' in str(self.hazy_root_dir)) or ('SOTS' in str(self.hazy_root_dir)):
            img_num = int(self.hazy_img_list[index].split('_')[0])
            hazy_img_path = self.hazy_img_list[index]
            gt_img_path = self.gt_img_dict[img_num]
        else :
            gt_img_path = self.gt_img_list[index]
            hazy_img_path = self.hazy_img_list[index]
        
        hazy_img = Image.open(Path(self.hazy_root_dir)/hazy_img_path).convert('RGB')
        gt_img = Image.open(Path(self.gt_root_dir)/gt_img_path).convert('RGB')
        hazy_img = np.asarray(hazy_img, np.float64) 
        gt_img = np.asarray(gt_img, np.float64)
        state = torch.get_rng_state()
        hazy_img = self.transform(hazy_img.astype(np.uint8)) 
        torch.set_rng_state(state)
        gt_img = self.transform(gt_img.astype(np.uint8))         
        if not self.for_test:
            t_gt = makeDarkChannel((255*gt_img).permute(1,2,0).contiguous().numpy()).transpose((2, 0, 1))    
        else :
            t_gt = -1
            
        if(self.mask == False):
            return {"hazy_img" : hazy_img, "gt_img" : gt_img, "t_gt": t_gt}
        elif(self.mask == True):
            temp_hazy_img = hazy_img.numpy()
            temp_hazy_img = temp_hazy_img.transpose((1, 2, 0))
            temp_hazy_img = temp_hazy_img * 0.5 + 0.5
            temp_hazy_img = np.clip(temp_hazy_img, 0, 1)
            temp_hazy_img *= 255.0 
            
            density_map = makeDensityMask(temp_hazy_img) 
            
            trans = transforms.ToTensor()
            density_map = trans(density_map) 
            
            hazy_img_concat = torch.cat((hazy_img, density_map), 0)
            
            return {"hazy_img_concat" : hazy_img_concat, "hazy_img" : hazy_img, "gt_img" : gt_img, "t_gt": t_gt}
            
        else: 
            print("WRONG MASK SETTING")

def sortedFileList(directory):
    list = os.listdir(directory)
    try:
        list.sort(key=lambda x : int(x.split('_')[0]))
    except:
        list.sort(key=lambda x : int(x.split('.')[0]))
    return list

def makeDarkChannel(img):
    """_summary_
    Args:
        img (numpy.ndarray) : [H, W, 3] (0 ~ 255)

    Returns:
        mask (numpy.ndarray) : [H, W, 1] (0 ~ 1)
    """
    I = img / 255.0
    dark = dcp.DarkChannel(I, 15)
    dark = dark[:, :, np.newaxis] 
    return dark

def makeDensityMask(img):
    """_summary_
    Args:
        img (numpy.ndarray) : [H, W, 3] (0 ~ 255)

    Returns:
        mask (numpy.ndarray) : [H, W, 1] (0 ~ 1)
    """
    I = img / 255.0
    dark = dcp.DarkChannel(I, 15)
    dark = dark[:, :, np.newaxis] 
    A = dcp.AtmLight(I, dark)
    t1 = dcp.TransmissionEstimate(I, A, 15)
    t2 = dcp.TransmissionRefine((I * 255.0).astype('float32'), t1)
    density_map = 1 - t2
    np.expand_dims(density_map, axis=2)
    density_map = density_map[:, :, np.newaxis] 
    return density_map

def makeDensityMaskTensor(img):
    """_summary_
    Args:
        img (torch.Tensor) : [B, 3, H, W] (normalized)

    Returns:
        mask (numpy.ndarray) : [B, 1, H, W] (0 ~ 1)
    """
    temp_output = []
    img = img.detach().cpu().numpy() 
    for i in range(img.shape[0]):
        crt_img = img[i, :, :, :].squeeze() 
        crt_img = crt_img.transpose((1, 2, 0)) 
        crt_img = crt_img * 0.5 + 0.5
        crt_img = np.clip(crt_img, 0, 1)
        crt_img *= 255.0
        
        crt_img_mask = makeDensityMask(crt_img) 
        crt_img_mask = crt_img_mask.transpose((2, 0, 1)) 
        crt_img_mask = crt_img_mask[np.newaxis, :, :, :] 
        
        temp_output.append(crt_img_mask)
    output = np.concatenate(temp_output, axis = 0) 
    density_map = torch.from_numpy(output)
    return density_map