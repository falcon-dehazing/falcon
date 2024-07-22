import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def peak_signal_noise_ratio_(gt, dehazed, data_range = 1.0):
    dehazed = dehazed.copy()
    
    single_psnr = peak_signal_noise_ratio(gt, dehazed, data_range = 1.0)
            
    psnr_result = single_psnr
    return psnr_result

def structural_similarity_(gt, dehazed, data_range = 1.0, channel_axis = 2):
    dehazed = dehazed.copy()
    
    single_ssim = structural_similarity(gt, dehazed, data_range = 1.0, channel_axis = 2)
    
    ssim_result = single_ssim
    return ssim_result    