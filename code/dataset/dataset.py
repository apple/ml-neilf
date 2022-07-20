#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
import os
from glob import glob
import torch
import numpy as np
import cv2

from utils import io, geometry

class NeILFDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,
                 data_folder,
                 validation_indexes,
                 num_pixel_samples,
                 mode='train'
                 ):

        assert os.path.exists(data_folder), "Data directory is empty"
        print ('NeILFDataset: loading data from: ' + data_folder)

        self.data_folder = data_folder
        self.num_pixel_samples = num_pixel_samples
        self.mode = mode
        self.use_brdf_gt = False
        self.use_depth_map = False

        # load cameras
        self.intrinsics, self.extrinsics, self.scale_mat, self.image_list, self.image_indexes, self.image_resolution = \
            io.load_cams_from_sfmscene(f'{self.data_folder}/inputs/sfm_scene.json')
        self.total_pixels = self.image_resolution[0] * self.image_resolution[1]

        # # test
        # self.image_indexes = self.image_indexes[0:3]
        # validation_indexes = [0, 1]

        # split training/validataion sets
        self.num_images = len(self.image_indexes)
        validation_list_indexes = [v % self.num_images for v in validation_indexes]
        self.validation_indexes = []
        self.training_indexes = []
        for i in range(self.num_images):
            image_index = self.image_indexes[i]
            if i in validation_list_indexes:
                self.validation_indexes.append(image_index)
            else:
                self.training_indexes.append(image_index)
        self.num_validation_images = len(self.validation_indexes)
        self.num_training_images = len(self.training_indexes)

        # uv coordinate
        uv = np.mgrid[0:self.image_resolution[0], 0:self.image_resolution[1]]
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        self.uv = uv.reshape(2, -1).transpose(1, 0)                                             # [HW, 2]

        # load and prepare validation data
        self._load_and_prepare_validation_data()

        # load and prepare training data
        if mode == 'train':
            self._load_and_prepare_training_data()

        # load positional texture atlas for exporting BRDF as texture maps
        if mode == 'eval':
            texture_folder = os.path.join(self.data_folder, 'inputs/model/pos_tex')
            pos_tex_files = glob(os.path.join(texture_folder, '*.exr'))
            self.tex_coords = []
            for pos_tex_file in pos_tex_files:
                coord, coord_mask = io.load_tex_coord(pos_tex_file)                             # [H, W, 3], [H, W]
                coord_shape = coord.shape[:2]
                coord = torch.from_numpy(coord[coord_mask]).float()                             # [N_v, 3]
                # normalize the coordinate system
                inv_scale_mat = torch.from_numpy(self.scale_mat).float().inverse()              # [4, 4]
                inv_scale_mat = inv_scale_mat.unsqueeze(0)                                      # [1, 4, 4]
                homo_coord = geometry.append_hom(coord, -1).unsqueeze(-1)                       # [N_v, 4, 1]
                coord = (inv_scale_mat @ homo_coord)[:, :3, 0]                                  # [N_v, 3]
                obj_name = (os.path.splitext(os.path.basename(pos_tex_file))[0]).split('_')[0]
                self.tex_coords.append((obj_name, coord, coord_shape, coord_mask))

    def _load_and_prepare_validation_data(self):
        
        print ('NeILFDataset: loading validation data')

        # load validation views
        self._load_data_from_indexes(self.validation_indexes)

        # prepare validation data
        self.validation_data = []
        for list_index in range(self.num_validation_images):

            validation_sample = {
                'intrinsics': self.all_intrinsics[list_index],                                  # [1, 4, 4]
                'pose': self.all_poses[list_index],                                             # [1, 4, 4]
                'uv': self.uv,                                                                  # [HW, 2]
                'positions': self.all_position_pixels[list_index].squeeze(0),                   # [HW, 3]
                'normals': self.all_normal_pixels[list_index].squeeze(0)                        # [HW, 3]
            }

            validation_ground_truth = {
                'rgb': self.all_rgb_pixels[list_index].squeeze(0)                               # [HW, 3]
            }
            if self.use_brdf_gt:
                validation_ground_truth['base_color'] = self.all_basecolor_maps[list_index]     # [H, W, 3]        
                validation_ground_truth['roughness'] = self.all_roughness_maps[list_index]      # [H, W, 1]
                validation_ground_truth['metallic'] = self.all_metallic_maps[list_index]        # [H, W, 1]

            self.validation_data.append(tuple([validation_sample, validation_ground_truth]))
        self.validation_data = self.collate_fn(self.validation_data)
        return 

    def _load_and_prepare_training_data(self):

        print ('NeILFDataset: loading training data')

        # load validation views
        self._load_data_from_indexes(self.training_indexes)

        # prepare training data by flatten to 1D lists         
        self.all_intrinsics = torch.cat(self.all_intrinsics, axis=0)                            # [T, 4, 4]
        self.all_poses = torch.cat(self.all_poses, axis=0)                                      # [T, 4, 4]

        self.train_rgb_pixels = torch.cat(self.all_rgb_pixels, axis=0)                          # [T, HW, 3]
        self.train_rgb_grad_pixels = torch.cat(self.all_rgb_grad_pixels, axis=0)                # [T, HW]
        self.train_position_pixels = torch.cat(self.all_position_pixels, axis=0)                # [T, HW, 3]
        self.train_normal_pixels = torch.cat(self.all_normal_pixels, axis=0)                    # [T, HW, 3]
        self.train_list_index_pixels = torch.cat(self.all_list_index_pixels, axis=0)            # [T, HW]
        self.train_rgb_pixels = self.train_rgb_pixels.reshape([-1, 3])                          # [THW, 3]
        self.train_rgb_grad_pixels = self.train_rgb_grad_pixels.reshape([-1])                   # [THW]
        self.train_position_pixels = self.train_position_pixels.reshape([-1, 3])                # [THW, 3]
        self.train_normal_pixels = self.train_normal_pixels.reshape([-1, 3])                    # [THW, 3]
        self.train_list_index_pixels = self.train_list_index_pixels.reshape([-1])               # [THW]
        self.train_uv = self.uv.repeat([self.num_training_images, 1])                           # [THW, 2]

        self.num_all_training_pixels = (self.train_list_index_pixels).shape[0]

        # permute all training pixels
        self._random_permute_all_training_pixels()

        return

    def _load_data_from_indexes(self, indexes_to_read):

        self.all_intrinsics = []
        self.all_poses = []
        self.all_rgb_pixels = []
        self.all_position_pixels = []
        self.all_normal_pixels = []
        self.all_list_index_pixels = []
        self.all_rgb_grad_pixels = []
        self.all_basecolor_maps = []
        self.all_roughness_maps = []
        self.all_metallic_maps = []
        
        # load all pixels
        for list_index, image_index in enumerate(indexes_to_read):

            # paths
            prefix = os.path.split(os.path.splitext(self.image_list[image_index])[0])[1]
            input_folder = os.path.join(self.data_folder, 'inputs')

            # only use depth map if there are depth maps but no position maps
            if not os.path.exists(os.path.join(input_folder, 'position_maps')) \
                and os.path.exists(os.path.join(input_folder, 'depth_maps')):
                geometry_type = 'depth_maps'
                self.use_depth_map = True

            geometry_type = 'depth_maps' if self.use_depth_map else 'position_maps'
            rgb_image_prefix = os.path.join(input_folder, 'images', prefix)
            position_map_prefix = os.path.join(input_folder, geometry_type, prefix)
            normal_map_prefix = os.path.join(input_folder, 'normal_maps', prefix)

            gt_brdf_folder = os.path.join(self.data_folder, 'ground_truths/materials')
            basecolor_map_path = os.path.join(gt_brdf_folder, 'kd', prefix + '.png')
            roughness_map_path = os.path.join(gt_brdf_folder, 'roughness', prefix + '.png')
            metallic_map_path = os.path.join(gt_brdf_folder, 'metallic', prefix + '.png')

            # only use brdf gt if there are brdf gts and in eval mode
            if os.path.isdir(os.path.join(gt_brdf_folder, 'kd')) and \
                os.path.isdir(os.path.join(gt_brdf_folder, 'roughness')) and \
                os.path.isdir(os.path.join(gt_brdf_folder, 'metallic')) and \
                self.mode == 'eval':
                self.use_brdf_gt = True

            # read input images, depth/position maps, and normal maps
            rgb_image = io.load_rgb_image_with_prefix(rgb_image_prefix)                         # [H, W, 3]
            rgb_gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)                              # [H, W]
            rgb_grad_x = cv2.Sobel(rgb_gray, cv2.CV_32F, 1, 0, ksize=cv2.FILTER_SCHARR)         # [H, W]
            rgb_grad_y = cv2.Sobel(rgb_gray, cv2.CV_32F, 0, 1, ksize=cv2.FILTER_SCHARR)         # [H, W]
            rgb_grad = cv2.magnitude(rgb_grad_x, rgb_grad_y)                                    # [H, W]
            if self.use_depth_map:
                depth_map = io.load_gray_image_with_prefix(position_map_prefix)                 # [H, W]
                position_map = geometry.depth_map_to_position_map(
                    depth_map, self.uv, 
                    self.extrinsics[image_index], self.intrinsics[image_index])                 # [H, W, 3], [H, W, 1]
            else:
                position_map = io.load_rgb_image_with_prefix(position_map_prefix)               # [H, W, 3]
            normal_map = io.load_rgb_image_with_prefix(normal_map_prefix)                       # [H, W, 3]

            # read BRDF ground truth for evaluation
            if self.use_brdf_gt:
                base_color = io.load_rgb_image(basecolor_map_path)                              # [H, W, 3]
                roughness = io.load_gray_image(roughness_map_path)                              # [H, W]
                roughness = roughness / 256                                                     # [H, W]
                metallic = io.load_gray_image(metallic_map_path)                                # [H, W]
                metallic = metallic / 256                                                       # [H, W]

            # apply scale mat to camera
            projection = self.intrinsics[image_index] @ self.extrinsics[image_index]            # [4, 4]
            scaled_projection = (projection @ self.scale_mat)[0:3, 0:4]                         # [3, 4]    
            intrinsic, pose = geometry.decompose_projection_matrix(scaled_projection)           # [4, 4], [4, 4]
            self.all_intrinsics.append(torch.from_numpy(intrinsic).float().unsqueeze(0))        # [N][1, 4, 4]             
            self.all_poses.append(torch.from_numpy(pose).float().unsqueeze(0))                  # [N][1, 4, 4]

            # apply scale mat to position map
            mask_map = (position_map != 0).astype(np.float)                                     # [H, W, 1]
            mask_map = np.sum(mask_map, axis=-1, keepdims=True)                                 # [H, W, 1]
            mask_map = (mask_map > 0).astype(np.float)                                          # [H, W, 1]
            position_map = geometry.append_hom(torch.from_numpy(position_map).float(), -1)      # [H, W, 3] -> [H, W, 4]  
            position_map = position_map.unsqueeze(-1)                                           # [H, W, 4, 1]
            inv_scale_mat = (torch.from_numpy(self.scale_mat).float()).inverse()                # [4, 4]
            inv_scale_mat = inv_scale_mat.unsqueeze(0).unsqueeze(0)                             # [1, 1, 4, 4]
            position_map = (inv_scale_mat @ position_map)[:, :, 0:3, 0]
            mask_map = torch.from_numpy(mask_map).float()                                       # [H, W, 1]
            position_map = position_map * mask_map                                              # [H, W, 3]

            # to tensors and reshape to pixels
            rgb_pixels = torch.from_numpy(rgb_image.reshape(1, -1, 3)).float()                  # [1, HW, 3]
            rgb_grad_pixels = torch.from_numpy(rgb_grad.reshape(1, -1)).float()                 # [1, HW]
            position_pixels = position_map.reshape(1, -1, 3)                                    # [1, HW, 3]  
            normal_pixels = torch.from_numpy(normal_map.reshape(1, -1, 3)).float()              # [1, HW, 3]
            list_index_pixels = torch.ones_like(position_pixels[0:1, :, 0]) * list_index        # [1, HW]
            
            # to tensors
            if self.use_brdf_gt:
                basecolor_pixels = torch.from_numpy(base_color).float()                         # [H, W, 3]
                roughness_pixels = torch.from_numpy(roughness).float()                          # [H, W, 1]
                metallic_pixels = torch.from_numpy(metallic).float()                            # [H, W, 1]

            # add to list 
            self.all_rgb_pixels.append(rgb_pixels)                                              # [N][1, HW, 3]
            self.all_rgb_grad_pixels.append(rgb_grad_pixels)                                    # [N][1, HW]
            self.all_position_pixels.append(position_pixels)                                    # [N][1, HW, 3]
            self.all_normal_pixels.append(normal_pixels)                                        # [N][1, HW, 3]
            self.all_list_index_pixels.append(list_index_pixels)                                # [N][1, HW]
            if self.use_brdf_gt:
                self.all_basecolor_maps.append(basecolor_pixels)                                # [N][H, W, 3]
                self.all_roughness_maps.append(roughness_pixels)                                # [N][H, W]
                self.all_metallic_maps.append(metallic_pixels)                                  # [N][H, W]

    def _random_permute_all_training_pixels(self):
        
        rand_indexes = torch.randperm(self.num_all_training_pixels)
        self.train_rgb_pixels = self.train_rgb_pixels[rand_indexes]
        self.train_rgb_grad_pixels = self.train_rgb_grad_pixels[rand_indexes]
        self.train_position_pixels = self.train_position_pixels[rand_indexes]
        self.train_normal_pixels = self.train_normal_pixels[rand_indexes]
        self.train_list_index_pixels = self.train_list_index_pixels[rand_indexes]
        self.train_uv = self.train_uv[rand_indexes]       
        self.index_in_rand = 0
        return

    def __len__(self):
        # a unbounded length
        return 10000000             

    def __getitem__(self, index_not_used):

        start = self.index_in_rand
        end = self.index_in_rand + self.num_pixel_samples

        rgb_batch = self.train_rgb_pixels[start:end, :]                                         # [B, 3]
        rgb_grad_batch = self.train_rgb_grad_pixels[start:end]                                  # [B]
        position_batch = self.train_position_pixels[start:end, :]                               # [B, 3]
        normal_batch = self.train_normal_pixels[start:end, :]                                   # [B, 3]
        uv_batch = self.train_uv[start:end]                                                     # [B, 2]
        index_batch = self.train_list_index_pixels[start:end].long()                            # [B]
        intrinsic_batch = self.all_intrinsics[index_batch]                                      # [B, 4, 4]
        pose_batch = self.all_poses[index_batch]                                                # [B, 4, 4]
        
        # permute all pixels only after traversing all
        self.index_in_rand += self.num_pixel_samples
        if self.index_in_rand > self.num_all_training_pixels:
            self._random_permute_all_training_pixels()

        sample = {
            'intrinsics': intrinsic_batch,
            'pose': pose_batch,
            'uv': uv_batch,
            'positions': position_batch,
            'normals': normal_batch
        }
        ground_truth = {
            'rgb': rgb_batch,
            'rgb_grad': rgb_grad_batch
        }
        
        return sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)