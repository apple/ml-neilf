#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
import torch
import torch.nn as nn
import torch.nn.functional as Func
import numpy as np

from utils import geometry
from model.ray_sampling import fibonacci_sphere_sampling
from model.nn_arch import NeILFMLP, BRDFMLP

EPS = 1e-7

class NeILFPBR(nn.Module):

    def __init__(self, 
                 brdf_nn, 
                 neilf_nn, 
                 num_train_incident_samples, 
                 num_eval_incident_samples):
        super().__init__()
        self.brdf_nn = brdf_nn
        self.neilf_nn = neilf_nn
        self.S = num_train_incident_samples
        self.S_evel = num_eval_incident_samples

    def sample_brdfs(self, points, is_gradient=False):

        points.requires_grad_(True)

        x = self.brdf_nn(points)                                                            # [N, 5]
        # convert from [-1,1] to [0,1]
        x = x / 2 + 0.5                                                                     # [N, 5]
        base_color, roughness, metallic = x.split((3, 1, 1), dim=-1)                        # [N, 3], [N, 1], [N, 1]
        
        # gradients w.r.t. input position
        gradients = []
        for brdf_slice in [roughness, metallic]:
            if is_gradient:
                d_output = torch.ones_like(brdf_slice, requires_grad=False)
                gradient = torch.autograd.grad(
                    outputs=brdf_slice,
                    inputs=points,
                    grad_outputs=d_output,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                    allow_unused=True)[0]
            else:
                gradient = torch.zeros_like(points)
            gradients.append(gradient)
        gradients = torch.stack(gradients, dim=-2)                                          # [N, 2, 3]

        return base_color, roughness, metallic, gradients

    def sample_incident_rays(self, normals, is_training):
        
        if is_training:
            incident_dirs, incident_areas = fibonacci_sphere_sampling(
                normals, self.S, random_rotate=True)
        else:
            incident_dirs, incident_areas = fibonacci_sphere_sampling(
                normals, self.S_evel, random_rotate=False)

        return incident_dirs, incident_areas                                                # [N, S, 3], [N, S, 1]

    def sample_incident_lights(self, points, incident_dirs):

        # point: [N, 3], incident_dirs: [N, S, 3]
        N, S, _ = incident_dirs.shape
        points = points.unsqueeze(1).repeat([1, S, 1])                                      # [N, S, 3]
        nn_inputs = torch.cat([points, -incident_dirs], axis=2)                             # [N, S, 6]
        nn_inputs = nn_inputs.reshape([-1, 6])                                              # [N * S, 6]
        incident_lights = self.neilf_nn(nn_inputs)                                          # [N * S, 3]
        # use mono light
        if incident_lights.shape[1] == 1:
            incident_lights = incident_lights.repeat([1, 3])
        return incident_lights.reshape([N, S, 3])                                           # [N, S, 3]

    def plot_point_env(self, point, width):

        def equirectangular_proj(width: int, meridian=0):
            height = width // 2
            x = height - torch.arange(width).float().repeat([height, 1])
            y = torch.arange(height).float().repeat(width, 1).t() - height // 2
            r = width / (2 * np.pi)
            lat = y / r
            lon = x / r + meridian
            coord = torch.stack([np.pi / 2 + lat, lon], dim=-1)
            valid_mask = torch.ones_like(lat).bool().unsqueeze(-1)
            return coord, valid_mask
            
        def sph2cart(sph, dim=-1):
            theta, phi = sph.split(1, dim=dim)
            x = theta.sin() * phi.cos()
            y = theta.sin() * phi.sin()
            z = theta.cos()
            cart = torch.cat([x,y,z], dim=dim)
            return cart

        # incident direction of all pixels in the env map
        eval_sph, valid_mask = equirectangular_proj(width, meridian=0)                      # [H, W, 2]
        eval_sph = eval_sph.to(point.device)
        valid_mask = valid_mask.to(point.device)
        eval_cart = sph2cart(eval_sph, dim=-1)                                              # [H, W, 3]
        eval_cart_flat = -1 * eval_cart.view([-1, 3])                                       # [N, 3]

        point = point.unsqueeze(0).repeat([eval_cart_flat.shape[0], 1])                     # [N, 3]
        nn_inputs = torch.cat([point, eval_cart_flat], axis=1)                              # [N, 6]
        env_map = self.neilf_nn(nn_inputs).view(-1, width, 3)                               # [N, 3]

        env_map = env_map * valid_mask
        return env_map

    def rendering_equation(self, 
                           output_dirs, 
                           normals, 
                           base_color, 
                           roughness, 
                           metallic, 
                           incident_lights, 
                           incident_dirs, 
                           incident_areas):

        # extend all inputs into shape [N, 1, 1/3] for multiple incident lights
        output_dirs = output_dirs.unsqueeze(dim=1)                                          # [N, 1, 3]
        normal_dirs = normals.unsqueeze(dim=1)                                              # [N, 1, 3]
        base_color = base_color.unsqueeze(dim=1)                                            # [N, 1, 3]
        roughness = roughness.unsqueeze(dim=1)                                              # [N, 1, 1]
        metallic = metallic.unsqueeze(dim=1)                                                # [N, 1, 1]

        def _dot(a, b):
            return (a * b).sum(dim=-1, keepdim=True)                                        # [N, 1, 1]

        def _f_diffuse(h_d_o, n_d_i, n_d_o, base_color, metallic, roughness):

            return (1 - metallic) * base_color / np.pi                                      # [N, 1, 1]

        def _f_specular(h_d_n, h_d_o, n_d_i, n_d_o, base_color, roughness, metallic):

            # used in SG, wrongly normalized
            def _d_sg(r, cos):
                r2 = (r * r).clamp(min=EPS)
                amp = 1 / (r2 * np.pi)
                sharp = 2 / r2
                return amp * torch.exp(sharp * (cos - 1))
            D = _d_sg(roughness, h_d_n)

            # Fresnel term F
            F_0 = 0.04 * (1 - metallic) + base_color * metallic                             # [N, 1, 3]
            F = F_0 + (1.0 - F_0) * ((1.0 - h_d_o) ** 5)                                    # [N, S, 1]

            # geometry term V, we use V = G / (4 * cos * cos) here
            def _v_schlick_ggx(r, cos):
                r2 = ((1 + r) ** 2) / 8
                return 0.5 / (cos * (1 - r2) + r2).clamp(min=EPS)
            V = _v_schlick_ggx(roughness, n_d_i) * _v_schlick_ggx(roughness, n_d_o)     

            return D * F * V                                                                # [N, S, 1]

        # half vector and all cosines
        half_dirs = incident_dirs + output_dirs                                             # [N, S, 3]
        half_dirs = Func.normalize(half_dirs, dim=-1)                                       # [N, S, 3]
        h_d_n = _dot(half_dirs, normal_dirs).clamp(min=0)                                   # [N, S, 1]
        h_d_o = _dot(half_dirs, output_dirs).clamp(min=0)                                   # [N, S, 1]
        n_d_i = _dot(normal_dirs, incident_dirs).clamp(min=0)                               # [N, S, 1]
        n_d_o = _dot(normal_dirs, output_dirs).clamp(min=0)                                 # [N, 1, 1]

        f_d = _f_diffuse(h_d_o, n_d_i, n_d_o, base_color, metallic, roughness)              # [N, 1, 3]
        f_s = _f_specular(h_d_n, h_d_o, n_d_i, n_d_o, base_color, roughness, metallic)      # [N, S, 3]

        rgb = ((f_d + f_s) * incident_lights * incident_areas * n_d_i).mean(dim=1)          # [N, 3]

        return rgb

    def forward(self, points, normals, view_dirs, is_training):

        # get brdf
        base_color, roughness, metallic, gradients = self.sample_brdfs(
            points, is_gradient=is_training)                                                # [N, 3]

        # sample incident rays for the input point
        incident_dirs, incident_areas = self.sample_incident_rays(
            normals, is_training)                                                           # [N, S, 3], [N, S, 1]

        # sample incident lights for the input point
        incident_lights = self.sample_incident_lights(points, incident_dirs)                # [N, S, 3]

        # the rendering equation, first pass
        rgb = self.rendering_equation(
            view_dirs, normals, base_color, roughness, metallic, 
            incident_lights, incident_dirs, incident_areas)                                 # [N, 3]

        return rgb, base_color, roughness, metallic, gradients


class NeILFModel(nn.Module):

    def __init__(self, config_model):
        super().__init__()

        self.to_ldr = config_model['use_ldr_image']
        num_train_incident_samples = config_model['num_train_incident_samples']
        num_eval_incident_samples = config_model['num_eval_incident_samples']

        print ("Number of training incident samples: ", num_train_incident_samples)
        print ("Number of evaluation incident samples: ", num_eval_incident_samples)

        # implicit brdf
        brdf_config = config_model['brdf_network']
        self.brdf_nn = BRDFMLP(**brdf_config)

        # implicit incident light
        neilf_config = config_model['neilf_network']
        self.neilf_nn = NeILFMLP(**neilf_config)

        # neilf rendering
        self.neilf_pbr = NeILFPBR(self.brdf_nn, self.neilf_nn, \
            num_train_incident_samples, num_eval_incident_samples)

        # learnable gamma parameter to map HDR to LDR
        if self.to_ldr:
            self.gamma = nn.Parameter(torch.ones(1).float().cuda())

    def input_preprocessing(self, input):

        # parse model input
        intrinsics = input["intrinsics"].reshape([-1, 4, 4])                            # [N, 4, 4]
        pose = input["pose"].reshape([-1, 4, 4])                                        # [N, 4, 4] NOTE: idr pose is inverse of mvsnet extrinsic
        uv = input["uv"].reshape([-1, 2])                                               # [N, 2]
        points = input["positions"].reshape([-1, 3])                                    # [N]
        normals = Func.normalize(input["normals"].reshape([-1, 3]), dim=1)              # [N, 3]
        total_samples = uv.shape[0]

        # pixel index to image coord
        uv = uv + 0.5                                                                   # [N, 2]

        # get viewing directions
        ray_dirs, _ = geometry.get_camera_params(uv, pose, intrinsics)                  # [N, 3]
        view_dirs = -ray_dirs                                                           # [N, 3]

        # get mask
        render_masks = (points != 0).sum(-1) > 0                                        # [N]

        return points, normals, view_dirs, render_masks, total_samples

    def plot_point_env(self, point, width):
        return self.neilf_pbr.plot_point_env(point, width)

    def forward(self, input, is_training=False):

        # parse input
        points, normals, view_dirs, render_masks, total_samples \
            = self.input_preprocessing(input)
        
        # neilf rendering
        rgb, base_color, roughness, metallic, gradients = self.neilf_pbr(
            points[render_masks], normals[render_masks], 
            view_dirs[render_masks], is_training)

        # convert to ldr
        if self.to_ldr:
            rgb = rgb.clamp(EPS, 1)
            rgb = rgb ** self.gamma

        # demask
        masked_outputs = [rgb, base_color, roughness, metallic, gradients]
        outputs = []
        for masked_output in masked_outputs:
            output = torch.zeros((total_samples, *masked_output.shape[1:]), \
                dtype=masked_output.dtype, device=masked_output.device)
            output[render_masks] = masked_output
            outputs.append(output)
        rgb_values, base_color, roughness, metallic = outputs[:-1]

        output = {
            'points': points,
            'normals': normals,
            'render_masks': render_masks,
            'rgb_values': rgb_values,
            'base_color': base_color,
            'roughness': roughness,
            'metallic': metallic,
            
        }
        
        if is_training:
            output['brdf_grads'] = outputs[-1]
            if self.to_ldr:
                output['gamma'] = self.gamma
            
        return output
