#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
import os
import argparse
import sys
sys.path.append('../code')
import numpy as np
import tqdm

# time
from datetime import datetime
import pytz
TZ = pytz.timezone('Asia/Hong_Kong')

# torch
import torch

# neilf
from dataset.dataset import NeILFDataset
from model.neilf_brdf import NeILFModel
from model.loss import NeILFLoss
import utils.io as io
import utils.general as general

class NeILFTrainer():

    def _create_output_folders(self):

        # create brdf/lighting output folders
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now().astimezone(TZ))
        self.timestamp_folder = os.path.join(self.output_folder, self.timestamp)
        self.plots_folder = os.path.join(self.timestamp_folder, 'plots')
        general.mkdir_ifnotexists(self.output_folder)
        general.mkdir_ifnotexists(self.timestamp_folder)
        general.mkdir_ifnotexists(self.plots_folder)

        # create model checkpoint folders
        self.checkpoint_folder = os.path.join(self.timestamp_folder, 'checkpoints')
        general.mkdir_ifnotexists(self.checkpoint_folder)
        general.mkdir_ifnotexists(os.path.join(self.checkpoint_folder, 'ModelParameters'))
        general.mkdir_ifnotexists(os.path.join(self.checkpoint_folder, 'OptimizerParameters'))
        general.mkdir_ifnotexists(os.path.join(self.checkpoint_folder, 'SchedulerParameters'))

    def _create_optimizer(self):

        self.lr = self.config['train']['lr']
        self.lr_decay = self.config['train']['lr_decay']
        self.lr_decay_iters = self.config['train']['lr_decay_iters']
        self.use_ldr_image = self.config['model']['use_ldr_image']
        # NeILF and BRDF MLPs
        param_groups = [
            {'name': 'brdf_nn', 'params': self.model.brdf_nn.parameters(), 'lr': self.lr},
            {'name': 'neilf_nn', 'params': self.model.neilf_nn.parameters(), 'lr': self.lr}
        ]
        # learnable HDR-LDR gamma correction
        if self.use_ldr_image:
            self.lr_scaler = self.config['train']['lr_scaler']
            param_groups.append(
                {'name': 'gamma', 'params': self.model.gamma, 'lr': self.lr_scaler})
        self.optimizer = torch.optim.Adam(param_groups, lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, self.lr_decay_iters, gamma=self.lr_decay)

    def _load_checkpoint(self):

        # fine-tune or not
        if self.last_timestamp == 'latest':
            timestamps = os.listdir(self.output_folder) - self.timestamp
            last_timestamps = [t for t in timestamps if t != self.timestamp]
            if (len(last_timestamps)) == 0:
                self.last_timestamp = None
            else:
                self.last_timestamp = sorted(last_timestamps)[-1]

        # load pre-trained model
        if self.last_timestamp is not None:
            # paths
            prefix = str(self.last_checkpoint) + '.pth'
            last_checkpoint_folder = os.path.join(
                self.output_folder, self.last_timestamp, 'checkpoints')
            model_path = os.path.join(last_checkpoint_folder, 'ModelParameters', prefix)
            optim_path = os.path.join(last_checkpoint_folder, 'OptimizerParameters', prefix)
            sched_path = os.path.join(last_checkpoint_folder, 'SchedulerParameters', prefix)
            # load
            model_params = torch.load(model_path)
            optim_params = torch.load(optim_path)
            sched_params = torch.load(sched_path)
            # set model
            self.model.load_state_dict(model_params['model_state_dict'])
            self.start_iteration = model_params['iteration']
            self.optimizer.load_state_dict(optim_params['optimizer_state_dict'])
            self.scheduler.load_state_dict(sched_params['scheduler_state_dict'])
    
    def _save_checkpoint(self, iteration):
        prefix = str(iteration) + '.pth'
        torch.save(
            {'iteration': iteration, 'model_state_dict': self.model.state_dict()},
            os.path.join(self.checkpoint_folder, 'ModelParameters', prefix))
        torch.save(
            {'iteration': iteration, 'model_state_dict': self.model.state_dict()},
            os.path.join(self.checkpoint_folder, 'ModelParameters', 'latest.pth'))
        torch.save(
            {'iteration': iteration, 'optimizer_state_dict': self.optimizer.state_dict()},
            os.path.join(self.checkpoint_folder, 'OptimizerParameters', prefix))
        torch.save(
            {'iteration': iteration, 'optimizer_state_dict': self.optimizer.state_dict()},
            os.path.join(self.checkpoint_folder, 'OptimizerParameters', 'latest.pth'))
        torch.save(
            {'iteration': iteration, 'scheduler_state_dict': self.scheduler.state_dict()},
            os.path.join(self.checkpoint_folder, 'SchedulerParameters', prefix))
        torch.save(
            {'iteration': iteration, 'scheduler_state_dict': self.scheduler.state_dict()},
            os.path.join(self.checkpoint_folder, 'SchedulerParameters', 'latest.pth'))


    def __init__(self, 
                 input_folder, 
                 output_folder, 
                 config_path, 
                 is_continue, 
                 timestamp, 
                 checkpoint):

        self.input_folder = input_folder
        self.output_folder = output_folder
        self.is_continue = is_continue
        self.last_timestamp = timestamp
        self.last_checkpoint = checkpoint
        
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        # create output folders
        self._create_output_folders()

        # load config
        config = io.load_config(config_path)
        self.config = config

        # load input data and create dataset
        validation_indexes = config['eval']['validation_indexes']
        num_pixel_samples = config['train']['num_pixel_samples']
        self.dataset = NeILFDataset(
            input_folder, validation_indexes, num_pixel_samples, mode='train')

        # create training data loader
        self.train_dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=1, shuffle=True, drop_last=True, 
            collate_fn=self.dataset.collate_fn)
        
        # create model
        self.model = NeILFModel(config['model'])
        if torch.cuda.is_available():
            self.model.cuda()

        # create loss functions
        lambertian_weighting = config['train']['lambertian_weighting']
        smoothness_weighting = config['train']['smoothness_weighting']
        self.loss = NeILFLoss(lambertian_weighting, smoothness_weighting)
        
        # create optimizer and scheduler
        self._create_optimizer()

        # load pre-trained model
        self.start_iteration = 0
        if is_continue:
            self._load_checkpoint()

        self.num_pixel_samples = config['train']['num_pixel_samples']
        self.total_pixels = self.dataset.total_pixels
        self.image_resolution = self.dataset.image_resolution
        self.n_batches = len(self.train_dataloader)
        self.training_iterations = config['train']['training_iterations']
        self.plot_frequency = config['eval']['plot_frequency']
        self.save_frequency = config['eval']['save_frequency']

    def train(self):

        # progress bar
        progress_bar = tqdm.tqdm(
            range(self.start_iteration, self.training_iterations), dynamic_ncols=True)

        # self.validate(0)

        # training
        self.model.train()
        for iteration, (model_input, ground_truth) in enumerate(self.train_dataloader):
            
            # progress iteration, start from 1
            progress_iter = self.start_iteration + iteration + 1

            # transfer input to gpu
            for attr in ['intrinsics', 'pose', 'uv', 'positions', 'normals']:
                model_input[attr] = model_input[attr].cuda()

            # run network
            model_outputs = self.model(model_input, is_training=True)
            
            # compute loss
            loss = self.loss(model_outputs, ground_truth)

            # optimize 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # log message
            current_lr = self.scheduler.get_last_lr()[0]
            log_msg = f'Training loss = {loss.item():.4f}, lr = {current_lr:.2e}'
            if self.use_ldr_image:
                log_msg = log_msg + f', gamma = {self.model.gamma.item():.4f}'
            progress_bar.update(1)
            progress_bar.set_description(log_msg)
            if progress_iter > self.training_iterations:
                progress_bar.close()

            # save model
            if progress_iter % self.save_frequency == 0:
                self._save_checkpoint(iteration)

            # validate and save plot
            if progress_iter % self.plot_frequency == 0:
                self.model.eval()
                self.validate(progress_iter)
                self.model.train()

            # finish training
            self.scheduler.step()
            if progress_iter > self.training_iterations:
                break

    def validate(self, iteration):

        model_input, ground_truth = self.dataset.validation_data

        for attr in ['intrinsics', 'pose', 'uv', 'positions', 'normals']:
            model_input[attr] = model_input[attr].cuda()

        # split inputs
        split_inputs = general.split_neilf_input(model_input, self.total_pixels)

        # generate outputs
        split_outputs = []
        for split_input in split_inputs:
            with torch.no_grad():
                split_output = self.model(split_input, is_training=False)
            split_outputs.append(
                {k:split_output[k].detach().cpu() for k in 
                ['rgb_values', 'points', 'normals', 'base_color',
                'roughness', 'metallic', 'render_masks']})

        # merge output
        batch_size = ground_truth['rgb'].shape[0]
        model_outputs = general.merge_neilf_output(
            split_outputs, self.total_pixels, batch_size)

        # image size
        H = self.image_resolution[0]
        W = self.image_resolution[1]

        # rendered mask
        mask = model_outputs['render_masks'].reshape([batch_size, H, W, 1]).float()

        # estimated image
        rgb_eval = model_outputs['rgb_values']
        rgb_eval = rgb_eval.reshape([batch_size, H, W, 3])
        if not self.use_ldr_image: 
            rgb_eval = general.hdr2ldr(rgb_eval)
        rgb_eval = rgb_eval * mask + (1 - mask)

        # gt image
        rgb_gt = ground_truth['rgb']
        rgb_gt = rgb_gt.reshape([batch_size, H, W, 3])
        if not self.use_ldr_image: 
            rgb_gt = general.hdr2ldr(rgb_gt)
        rgb_gt = rgb_gt * mask + (1 - mask)

        # estimated BRDF
        base_eval = model_outputs['base_color'].reshape([batch_size, H, W, 3])    
        base_eval = base_eval * mask + (1 - mask)
        roug_eval = model_outputs['roughness'].repeat([1, 3])
        roug_eval = roug_eval.reshape([batch_size, H, W, 3])    
        meta_eval = model_outputs['metallic'].repeat([1, 3])
        meta_eval = meta_eval.reshape([batch_size, H, W, 3])

        # create figure to plot
        rgb_plot = torch.cat([rgb_eval, rgb_gt, base_eval, roug_eval, meta_eval], dim=1)    # [V, H * 5, W, 3]
        rgb_plot = rgb_plot.permute([0, 2, 1, 3])                                           # [V, W, H * 5, 3]
        rgb_plot = rgb_plot.reshape([-1, rgb_plot.shape[2], rgb_plot.shape[3]])             # [V * W, 5 * H, 3]
        rgb_plot = rgb_plot.permute([1, 0, 2])                                              # [5 * H, V * W, 3]
        rgb_plot = (rgb_plot.clamp(0, 1).detach().numpy() * 255).astype(np.uint8)           # [5 * H, V * W, 3]

        # save figure to file
        io.save_image(f'{self.plots_folder}/render_{iteration}.jpg', rgb_plot)                    

        # calculate PSNR
        psnr = 0
        for i in range(batch_size):
            psnr += general.calculate_psnr(rgb_gt[i].numpy(), rgb_eval[i].numpy())
        psnr /= batch_size
        print (f'Validation at iteration: {iteration}, PSNR: {psnr.item():.4f}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('input_folder', type=str, 
                        help='Input folder of images, cameras and geometry files.')
    parser.add_argument('output_folder', type=str, 
                        help='Output folder for saving trained models and results')
    parser.add_argument('--config_path', type=str, 
                        default='./confs/synthetic_neilf_brdf.json')    
    # finetuning options
    parser.add_argument('--is_continue', default=False, action="store_true", 
                        help='If set, indicates fine-tuning from a previous run.')
    parser.add_argument('--timestamp', default='latest', type=str, 
                        help='The timestamp to be used if from a previous run.')
    parser.add_argument('--checkpoint', default='latest', type=str,
                        help='The checkpoint to be used if from a previous run.')
    args = parser.parse_args()

    # run training
    trainer = NeILFTrainer(input_folder=args.input_folder,
                           output_folder=args.output_folder,
                           config_path=args.config_path,
                           is_continue=args.is_continue,
                           timestamp=args.timestamp,
                           checkpoint=args.checkpoint)
    trainer.train()