#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
import sys
sys.path.append('../code')
import numpy as np
from math import sqrt

import torch
import torch.nn.functional as Func

from utils import geometry

EPS = 1e-7

def fibonacci_sphere_sampling(normals, sample_num, random_rotate=True):

    N = normals.shape[0]
    delta = np.pi * (3.0 - np.sqrt(5.0))

    # fibonacci sphere sample around z axis 
    idx = torch.arange(sample_num).cuda().float().unsqueeze(0).repeat([N, 1])   # [N, S]
    z = 1 - 2 * idx / (2 * sample_num - 1)                                      # [N, S]
    rad = torch.sqrt(1 - z ** 2)                                                # [N, S]
    theta = delta * idx                                                         # [N, S]        
    if random_rotate:
        theta = torch.rand(N, 1).cuda() * 2 * np.pi + theta                     # [N, S]
    y = torch.cos(theta) * rad                                                  # [N, S]
    x = torch.sin(theta) * rad                                                  # [N, S]
    z_samples = torch.stack([x, y, z], axis=-1).permute([0, 2, 1])              # [N, 3, S]

    # rotate to normal
    z_vector = torch.zeros_like(normals)                                        # [N, 3]
    z_vector[:, 2] = 1                                                          # [N, 3]
    rotation_matrix = geometry.rotation_between_vectors(z_vector, normals)      # [N, 3, 3]
    incident_dirs = rotation_matrix @ z_samples                                 # [N, 3, S]
    incident_dirs = Func.normalize(incident_dirs, dim=1).permute([0, 2, 1])     # [N, S, 3]
    incident_areas = torch.ones_like(incident_dirs)[..., 0:1] * 2 * np.pi       # [N, S, 1]

    return incident_dirs, incident_areas