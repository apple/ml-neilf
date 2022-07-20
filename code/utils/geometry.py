#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
import numpy as np
import cv2
import torch
from torch.nn import functional as F

EPS = 1e-7

def decompose_projection_matrix(P):
    ''' Decompose intrincis and extrinsics from projection matrix (for Numpt object) '''
    # For Numpy object

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose

def get_camera_params(uv, pose, intrinsics):
    ''' Project an image to rays (for Tensor object) '''

    cam_loc = pose[:, 0:3, 3]                                                       # [N, 3]
    p = pose                                                                        # [N, 4, 4]

    x_cam = uv[:, 0:1]                                                              # [N, 1]
    y_cam = uv[:, 1:2]                                                              # [N, 1]
    z_cam = torch.ones_like(uv[:, 0:1]).cuda()                                      # [N, 1]
    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics)             # [N, 4]

    world_coords = torch.bmm(p, pixel_points_cam)[:, 0:3, 0]                        # [N, 3]
    ray_dirs = world_coords - cam_loc
    ray_dirs = F.normalize(ray_dirs, dim=1)

    return ray_dirs, cam_loc                                                        # [N, 3], [N, 3]

def lift(x, y, z, intrinsics):
    ''' Project a point from image space to camera space (for Tensor object) '''
    # For Tensor object
    # parse intrinsics
    fx = intrinsics[:, 0, 0].unsqueeze(1)                                           # [N]
    fy = intrinsics[:, 1, 1].unsqueeze(1)                                           # [N]
    cx = intrinsics[:, 0, 2].unsqueeze(1)                                           # [N]
    cy = intrinsics[:, 1, 2].unsqueeze(1)                                           # [N]
    sk = intrinsics[:, 0, 1].unsqueeze(1)                                           # [N]

    x_lift = (x - cx + cy * sk / fy - sk * y / fy) / fx * z                         # [N]
    y_lift = (y - cy) / fy * z                                                      # [ N]

    # homogeneous coordinate
    return torch.stack([x_lift, y_lift, z, torch.ones_like(z).cuda()], dim=1)       # [N, 4, 1]

def rotation_between_vectors(vec1, vec2):
    ''' Retruns rotation matrix between two vectors (for Tensor object) '''
    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/476311#476311
    # vec1.shape = [N, 3]
    # vec2.shape = [N, 3]
    batch_size = vec1.shape[0]
    
    v = torch.cross(vec1, vec2)                                                     # [N, 3, 3]

    cos = torch.bmm(vec1.view(batch_size, 1, 3), vec2.view(batch_size, 3, 1))
    cos = cos.reshape(batch_size, 1, 1).repeat(1, 3, 3)                             # [N, 3, 3]
    
    skew_sym_mat = torch.zeros(batch_size, 3, 3).cuda()
    skew_sym_mat[:, 0, 1] = -v[:, 2]
    skew_sym_mat[:, 0, 2] = v[:, 1]
    skew_sym_mat[:, 1, 0] = v[:, 2]
    skew_sym_mat[:, 1, 2] = -v[:, 0]
    skew_sym_mat[:, 2, 0] = -v[:, 1]
    skew_sym_mat[:, 2, 1] = v[:, 0]

    identity_mat = torch.zeros(batch_size, 3, 3).cuda()
    identity_mat[:, 0, 0] = 1
    identity_mat[:, 1, 1] = 1
    identity_mat[:, 2, 2] = 1

    R = identity_mat + skew_sym_mat
    R = R + torch.bmm(skew_sym_mat, skew_sym_mat) / (1 + cos).clamp(min=1e-7)
    zero_cos_loc = (cos == -1).float()
    R_inverse = torch.zeros(batch_size, 3, 3).cuda()
    R_inverse[:, 0, 0] = -1
    R_inverse[:, 1, 1] = -1
    R_inverse[:, 2, 2] = -1
    R_out = R * (1 - zero_cos_loc) + R_inverse * zero_cos_loc                       # [N, 3, 3]

    return R_out        

def get_depth(points, pose):
    ''' Retruns depth from 3D points according to camera pose (for Tensor object) '''
    batch_size, num_samples, _ = points.shape

    points_hom = torch.cat(
        [points, torch.ones((batch_size, num_samples, 1)).cuda()], dim=2)

    # permute for batch matrix product
    points_hom = points_hom.permute(0, 2, 1)

    points_cam = torch.inverse(pose).bmm(points_hom)
    depth = points_cam[:, 2, :][:, :, None]
    return depth

def get_points(depths, coords, pose, intrinsic):
    ''' Return 3D points from depth value according to camera pose (for Tensor object) '''
    batch_size, num_samples = depths.shape
    
    inv_pose = torch.inverse(pose)
    rotation = inv_pose[:, :3, :3]
    translation = inv_pose[:, :3, 3]
    translation = translation.unsqueeze(2)
    intrinsic = intrinsic[:, 0:3, 0:3]

    coords_hom = torch.cat(
        (coords, torch.ones((batch_size, num_samples, 1)).cuda()), dim=2)
    coords_hom = coords_hom.permute(0, 2, 1)

    points_cam = torch.inverse(intrinsic).bmm(coords_hom)
    depths = depths.unsqueeze(2).permute(0, 2, 1)
    points_cam = depths * points_cam

    points = torch.inverse(rotation).bmm(points_cam - translation)
    points = points.permute(0, 2, 1)

    return points

def depth_map_to_position_map(depth_map, uv, extrinsic, intrinsic):
    ''' Return 3D points from depth value according to camera pose (for Numpy object) '''
    # depth_map: [H, W]
    # uv: [HW, 2]
    # extrinsic: [4, 4]
    # intrinsic: [4, 4]
    
    rotation = extrinsic[0:3, 0:3]                                          # [3, 3]
    inv_rotation = np.linalg.inv(rotation)                                  # [3, 3]
    translation = extrinsic[0:3, 3:4]                                       # [3, 1]
    intrinsic = intrinsic[0:3, 0:3]                                         # [3, 3]
    inv_intrinsic = np.linalg.inv(intrinsic)                                # [3, 3]

    H = depth_map.shape[0]
    W = depth_map.shape[1]
    HW = H * W
    depths = depth_map.reshape([HW])                                        # [HW]

    coords_hom = np.concatenate([uv + 0.5, np.ones([HW, 1])], 1)            # [HW, 3]
    coords_hom = np.transpose(coords_hom, [1, 0])                           # [3, HW]

    points_cam = inv_intrinsic @ coords_hom                                 # [3, HW]
    depths = np.expand_dims(depths, 0)                                      # [1, HW]
    points_cam = depths * points_cam                                        # [3, HW]

    points = inv_rotation @ (points_cam - translation)                      # [3, HW]
    points = np.transpose(points, [1, 0])                                   # [HW, 3]
    position_map = points.reshape([H, W, 3])                                # [H, W, 3]

    mask_map = np.expand_dims((depth_map > EPS).astype(np.float), 2)        # [H, W, 1]
    position_map = position_map * mask_map

    return position_map

def append_hom(arr, dim):
    shape = list(arr.shape)
    shape[dim] = 1
    append = torch.cat(
        [arr, torch.ones(*shape, dtype=arr.dtype, device=arr.device)], dim=dim)
    return append

def fib_sphere(m):
    # https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    phi = np.pi * (3.0 - np.sqrt(5.0))
    idx = np.arange(m).astype(np.float32)
    y = 1 - 2 * idx / (m-1)
    rad = np.sqrt(1 - y**2)
    theta = phi * idx
    x = np.cos(theta) * rad
    z = np.sin(theta) * rad
    cart = np.stack([x, y, z], axis=-1)       # [S, 3]
    cart = torch.from_numpy(cart).float()
    return cart 
