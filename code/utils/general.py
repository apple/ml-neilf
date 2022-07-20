#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
import os
import torch
import numpy as np
from skimage.metrics import structural_similarity as SSIM, peak_signal_noise_ratio as PSNR

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

def split_neilf_input(input, total_pixels):
    '''
     Split the input to fit Cuda memory for large resolution.
     Can decrease the value of split_num in case of cuda out of memory error.
     '''
    split_size = 1000
    split_input = []
    split_indexes = torch.split(torch.arange(total_pixels).cuda(), split_size, dim=0)
    for indexes in split_indexes:
        chunk_size = indexes.shape[0]
        data = {}
        data['uv'] = torch.index_select(input['uv'], 1, indexes)
        data['normals'] = torch.index_select(input['normals'], 1, indexes)
        data['positions'] = torch.index_select(input['positions'], 1, indexes)
        data['intrinsics'] = input['intrinsics'].repeat([1, chunk_size, 1, 1])
        data['pose'] = input['pose'].repeat([1, chunk_size, 1, 1])
        split_input.append(data)
    return split_input

def merge_neilf_output(res, total_pixels, batch_size):
    ''' Merge the split output. '''
    model_outputs = {}
    for entry in res[0]:
        if res[0][entry] is None:
            continue
        if len(res[0][entry].shape) == 1:
            model_outputs[entry] = torch.cat(
                [r[entry].reshape(batch_size, -1, 1) for r in res], 1
                ).reshape(batch_size * total_pixels)
        else:
            model_outputs[entry] = torch.cat(
                [r[entry].reshape(batch_size, -1, r[entry].shape[-1]) for r in res], 1
                ).reshape(batch_size * total_pixels, -1)
    return model_outputs

def calculate_ssim(img1, img2, mask=None):
    if mask is None: 
        mask = np.ones_like(img1)
    img1, img2 = [(arr * mask).astype(np.float64) for arr in [img1, img2]]
    return SSIM(img1, img2, multichannel=True, data_range=1.0)

def calculate_psnr(img1, img2, mask=None):
    if mask is None: 
        mask = np.ones_like(img1)
    img1, img2 = [(arr * mask).astype(np.float64) for arr in [img1, img2]]
    return PSNR(img1, img2, data_range=1.0)

def hdr2ldr(img, scale=0.666667):
    img = img * scale
    # img = 1 - np.exp(-3.0543 * img)  # Filmic
    img = (img * (2.51 * img + 0.03)) / (img * (2.43 * img + 0.59) + 0.14)  # ACES
    return img