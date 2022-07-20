#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
import argparse
import os
import torch
import numpy as np
import imageio
import json
from collections import defaultdict

import sys
sys.path.append('../code')
from dataset.dataset import NeILFDataset
from model.neilf_brdf import NeILFModel
from utils import general, io

def evaluate(input_data_folder,
             output_model_folder,
             config_path,
             timestamp,
             checkpoint,
             eval_nvs,
             eval_brdf,
             eval_lighting,
             export_nvs,
             export_brdf,
             export_lighting):

    assert os.path.exists(input_data_folder), "Data directory is empty"
    assert os.path.exists(output_model_folder), "Model directorty is empty"
    torch.set_default_dtype(torch.float32)

    # load config file
    config = io.load_config(config_path)

    # load input data and create evaluation dataset
    validation_indexes = config['eval']['validation_indexes']
    num_pixel_samples = config['train']['num_pixel_samples']
    eval_dataset = NeILFDataset(
        input_data_folder, validation_indexes, num_pixel_samples, mode='eval')

    # create model
    model = NeILFModel(config['model'])
    if torch.cuda.is_available():
        model.cuda()

    # load model
    if timestamp == 'latest':
        timestamps = os.listdir(output_model_folder)
        if len(timestamps) == 0:
            print('WRONG MODEL FOLDER')
            exit(-1)
        else:
            timestamp = sorted(timestamps)[-1]
    checkout_folder = os.path.join(output_model_folder, timestamp, 'checkpoints')
    prefix = str(checkpoint) + '.pth'
    model_path = os.path.join(checkout_folder, 'ModelParameters', prefix)
    model_params = torch.load(model_path)
    model.load_state_dict(model_params['model_state_dict'])

    # create evaluation folder
    eval_folder = os.path.join(output_model_folder, timestamp, 'evaluation')
    general.mkdir_ifnotexists(eval_folder)

    # evaluate BRDFs and novel view renderings
    if eval_brdf or eval_nvs:

        # results = dict()
        results = defaultdict(lambda: defaultdict(dict))

        # get validation data in the dataset
        model_input, ground_truth = eval_dataset.validation_data
        for attr in ['intrinsics', 'pose', 'uv', 'positions', 'normals']:
            model_input[attr] = model_input[attr].cuda()

        # split inputs
        total_pixels = eval_dataset.total_pixels
        split_inputs = general.split_neilf_input(model_input, total_pixels)

        # generate outputs
        split_outputs = []
        for split_input in split_inputs:
            with torch.no_grad():
                split_output = model(split_input, is_training=False)
            split_outputs.append(
                {k:split_output[k].detach().cpu() for k in 
                ['rgb_values', 'points', 'normals', 'base_color',
                'roughness', 'metallic', 'render_masks']})

        # merge output
        num_val_images = len(eval_dataset.validation_indexes)
        model_outputs = general.merge_neilf_output(
            split_outputs, total_pixels, num_val_images)

        # image size
        H = eval_dataset.image_resolution[0]
        W = eval_dataset.image_resolution[1]

        # rendered mask
        mask = model_outputs['render_masks']
        mask = mask.reshape([num_val_images, H, W, 1]).float()
        mask_np = mask.numpy()

        # estimated image
        rgb_eval = model_outputs['rgb_values']
        rgb_eval = rgb_eval.reshape([num_val_images, H, W, 3])
        if not config['model']['use_ldr_image']: 
            rgb_eval = general.hdr2ldr(rgb_eval)
        rgb_eval = rgb_eval * mask + (1 - mask)
        rgb_eval_np = rgb_eval.numpy()

        # gt image
        rgb_gt = ground_truth['rgb']
        rgb_gt = rgb_gt.reshape([num_val_images, H, W, 3])
        if not config['model']['use_ldr_image']: 
            rgb_gt = general.hdr2ldr(rgb_gt)
        rgb_gt = rgb_gt * mask + (1 - mask)
        rgb_gt_np = rgb_gt.numpy()

        # evaluate novel view renderings
        if eval_nvs:
            for i in range(num_val_images):
                index = str(eval_dataset.validation_indexes[i])
                results['render']['psnr'][index] = general.calculate_psnr(
                    rgb_gt_np[i], rgb_eval_np[i], mask_np[i])
                results['render']['ssim'][index] = general.calculate_ssim(
                    rgb_gt_np[i], rgb_eval_np[i], mask_np[i])

        # evaluate BRDFs
        if eval_brdf:

            # estimated BRDF
            base_eval = model_outputs['base_color'].reshape([num_val_images, H, W, 3])    
            base_eval = base_eval * mask + (1 - mask)
            roug_eval = model_outputs['roughness'].reshape([num_val_images, H, W, 1])    
            meta_eval = model_outputs['metallic'].reshape([num_val_images, H, W, 1])
            base_eval_np = base_eval.numpy()
            roug_eval_np = roug_eval.numpy()
            meta_eval_np = meta_eval.numpy()

            # gt BRDF
            base_gt_np = ground_truth['base_color'].reshape([num_val_images, H, W, 3])
            roug_gt_np = ground_truth['roughness'].reshape([num_val_images, H, W, 1])
            meta_gt_np = ground_truth['metallic'].reshape([num_val_images, H, W, 1])
            base_gt_np = base_gt_np.numpy()
            roug_gt_np = roug_gt_np.numpy()
            meta_gt_np = meta_gt_np.numpy()

            for i in range(num_val_images):
                index = str(eval_dataset.validation_indexes[i])
                # base color
                results['base_color']['psnr'][index] = general.calculate_psnr(
                    base_gt_np[i], base_eval_np[i], mask_np[i])
                results['base_color']['ssim'][index] = general.calculate_ssim(
                    base_gt_np[i], base_eval_np[i], mask_np[i])
                # roughness
                results['roughness']['psnr'][index] = general.calculate_psnr(
                    roug_gt_np[i], roug_eval_np[i], mask_np[i])
                results['roughness']['ssim'][index] = general.calculate_ssim(
                    roug_gt_np[i], roug_eval_np[i], mask_np[i])
                # metallic
                results['metallic']['psnr'][index] = general.calculate_psnr(
                    meta_gt_np[i], meta_eval_np[i], mask_np[i])
                results['metallic']['ssim'][index] = general.calculate_ssim(
                    meta_gt_np[i], meta_eval_np[i], mask_np[i])
    
        # calculate mean scores
        for item in results:
            for metric in results[item]:
                results[item][metric]['mean'] = 0
                for i in range(num_val_images):
                    index = str(eval_dataset.validation_indexes[i])
                    results[item][metric]['mean'] += results[item][metric][index]
                results[item][metric]['mean'] /= num_val_images
        
        # print results
        for item in results:
            print (item + ' evaluation:')
            for metric in results[item]:
                print ('  mean ' + metric + ': ' + str(results[item][metric]['mean']))

        # save results
        eval_report_path = os.path.join(eval_folder, 'report_evaluation.json')
        with open(eval_report_path, 'w') as eval_report:
            json.dump(results, eval_report, indent=4)

    # export BRDF as texture maps
    if export_brdf:
        for obj_name, tex_coord, coord_shape, coord_mask in eval_dataset.tex_coords:
            split_outputs = []
            batch_size = 10000
            for start in range(0, tex_coord.shape[0], batch_size):
                end = start + batch_size
                split_coord = tex_coord[start:end].cuda()
                brdf_output = model.neilf_pbr.sample_brdfs(split_coord)
                brdf_output = [arr.detach().cpu().numpy() for arr in brdf_output]
                split_outputs.append(brdf_output)
            valid_output = [np.concatenate(out, axis=0) for out in zip(*split_outputs)]
            for i, brdf_name in enumerate(['base_color', 'roughness', 'metallic']):
                out_map = np.zeros([coord_shape[0] * coord_shape[1], valid_output[i].shape[-1]])
                coord_mask = coord_mask.reshape([-1])
                out_map[coord_mask] = valid_output[i]
                out_map = out_map.reshape([*coord_shape, valid_output[i].shape[-1]])
                out_prefix = f'{eval_folder}/{obj_name}_{brdf_name}.png'
                imageio.imwrite(os.path.join(output_model_folder, out_prefix), out_map)
    
    # TODO: more evaluation options
    if eval_lighting: 
        print ('eval_lighting is not implemented yet')
    if export_lighting:
        print ('export_lighting is not implemented yet')
    if export_nvs:
        print ('export_nvs is not implemented yet')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('input_data_folder', type=str, 
                        help='input folder of images, cameras and geometry files.')
    parser.add_argument('output_model_folder', type=str, 
                        help='folder containing trained models, and for saving results')
    parser.add_argument('--config_path', type=str, 
                        default='./confs/synthetic_neilf_brdf.json')    

    # checkpoint
    parser.add_argument('--timestamp', default='latest',
                        type=str, help='the timestamp of the model to be evaluated.')
    parser.add_argument('--checkpoint', default='latest',
                        type=str, help='the checkpoint of the model to be evaluated')
    
    # items to evaluate
    parser.add_argument('--eval_nvs', action='store_true', default=False, 
                        help="evaluate novel view renderings")
    parser.add_argument('--eval_brdf', action='store_true', default=False, 
                        help="evaluate BRDF maps at novel views")
    parser.add_argument('--eval_lighting', action='store_true', default=False, 
                        help="work in progress, not ready yet")
    parser.add_argument('--export_nvs', action='store_true', default=False, 
                        help="export novel view renderings")
    parser.add_argument('--export_brdf', action='store_true', default=False, 
                        help="export BRDF as texture maps")
    parser.add_argument('--export_lighting', action='store_true', default=False, 
                        help="export incident lights at certain positions")

    args = parser.parse_args()
        
    evaluate(args.input_data_folder,
             args.output_model_folder,
             args.config_path,
             args.timestamp,
             args.checkpoint,
             args.eval_nvs,
             args.eval_brdf,
             args.eval_lighting,
             args.export_nvs,
             args.export_brdf,
             args.export_lighting)