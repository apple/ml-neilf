#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
import os
import numpy as np
import imageio
import skimage
import json

def load_gray_image(path):
    ''' Load gray scale image (both uint8 and float32) into image in range [0, 1] '''
    ext = os.path.splitext(path)[1]
    if ext in ['.png', '.jpg']:
        image = imageio.imread(path, as_gray=True)                      # [H, W]
    elif ext in ['.tiff', '.exr']:
        image = imageio.imread(path)                                    # [H, W]
        if len(image.shape) > 2:
            print ('Reading rgbfloat32 image as gray, will use the first channel')
            image = image[:, :, 0]
    image = skimage.img_as_float32(image)
    return image

def load_gray_image_with_prefix(prefix):
    ''' Load image using prefix to support different data type '''
    exts = ['.png', '.jpg', '.tiff', '.exr']
    for ext in exts:
        path = prefix + ext
        if os.path.exists(path):
            return load_gray_image(path)
    print ('Does not exists any image file with prefix: ' + prefix)
    return None

def load_rgb_image(path):
    ''' Load RGB image (both uint8 and float32) into image in range [0, 1] '''
    image = imageio.imread(path)[..., 0:3]                          # [H, W, 4] -> [H, W ,3]
    image = skimage.img_as_float32(image)
    return image


def load_rgb_image_with_prefix(prefix):
    ''' Load image using prefix to support different data type '''
    exts = ['.png', '.jpg', '.tiff', '.exr']
    for ext in exts:
        path = prefix + ext
        if os.path.exists(path):
            return load_rgb_image(path)
    print ('Does not exists any image file with prefix: ' + prefix)
    return None


def save_image(path, image):
    imageio.imwrite(path, image)

def load_mask_image(path):
    alpha = imageio.imread(path, as_gray=True)                      # [H, W]
    alpha = skimage.img_as_float32(alpha)
    object_mask = alpha > 127.5
    return object_mask

def load_tex_coord(path, mask=None):
    coord_image = imageio.imread(path)[..., 0:3]                    # [H, W, 4] -> [H, W ,3]
    if mask is not None:
        mask_image = imageio.imread(mask, as_gray=True) > 127.5
    else:
        mask_image = np.ones(coord_image.shape[:2], dtype=np.bool_)
    return coord_image, mask_image

def load_cams_from_sfmscene(path):

    # load json file
    with open(path) as f:
        sfm_scene = json.load(f)

    # camera parameters
    intrinsics = dict()
    extrinsics = dict()
    camera_info_list = sfm_scene['camera_track_map']['images']
    for i, (index, camera_info) in enumerate(camera_info_list.items()):
        # flg == 2 stands for valid camera 
        if camera_info['flg'] == 2:
            intrinsic = np.zeros((4, 4))
            intrinsic[0, 0] = camera_info['camera']['intrinsic']['focal'][0]
            intrinsic[1, 1] = camera_info['camera']['intrinsic']['focal'][1]
            intrinsic[0, 2] = camera_info['camera']['intrinsic']['ppt'][0]
            intrinsic[1, 2] = camera_info['camera']['intrinsic']['ppt'][1]
            intrinsic[2, 2] = intrinsic[3, 3] = 1
            extrinsic = np.array(camera_info['camera']['extrinsic']).reshape(4, 4)
            intrinsics[index] = intrinsic
            extrinsics[index] = extrinsic

    # load bbox transform
    bbox_transform = np.array(sfm_scene['bbox']['transform']).reshape(4, 4)

    # compute scale_mat for coordinate normalization
    scale_mat = bbox_transform.copy()
    scale_mat[[0,1,2],[0,1,2]] = scale_mat[[0,1,2],[0,1,2]].max() / 2
    
    # meta info
    image_list = sfm_scene['image_path']['file_paths']
    image_indexes = [str(k) for k in sorted([int(k) for k in image_list])]
    resolution = camera_info_list[image_indexes[0]]['size'][::-1]

    return intrinsics, extrinsics, scale_mat, image_list, image_indexes, resolution

def load_config(path):
    config_file = open(path)
    config = json.load(config_file)
    config_file.close()
    return config