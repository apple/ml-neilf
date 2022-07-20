#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
from torch import nn

class NeILFLoss(nn.Module):
    def __init__(self, lambertian_weighting, smoothness_weighting):
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction='sum')
        self.reg_weight = lambertian_weighting
        self.smooth_weight = smoothness_weighting

    def forward(self, model_outputs, ground_truth):
        
        masks = model_outputs['render_masks'].float()
        mask_sum = masks.sum().clamp(1e-7)

        # rendered rgb 
        rgb_gt = ground_truth['rgb'].cuda().reshape(-1, 3)
        rgb_values = model_outputs['rgb_values']
        rgb_loss = ((rgb_values - rgb_gt).abs() * masks.unsqueeze(1)).sum() / mask_sum

        # smoothness smoothness
        rgb_grad = ground_truth['rgb_grad'].cuda().reshape(-1)
        brdf_grads = model_outputs['brdf_grads']                # [N, 2, 3]
        smooth_loss = (brdf_grads.norm(dim=-1).sum(dim=-1) * (-rgb_grad).exp() * masks).mean()

        # lambertian assumption
        roughness = model_outputs['roughness']
        metallic = model_outputs['metallic']
        # reg_loss = ((roughness - 1).abs() * masks.unsqueeze(1)).sum() / mask_sum + \
        #     ((metallic - 0).abs() * masks.unsqueeze(1)).sum() / mask_sum
        reg_loss = ((metallic - 0).abs() * masks.unsqueeze(1)).sum() / mask_sum
    
        loss = rgb_loss + self.smooth_weight * smooth_loss + self.reg_weight * reg_loss

        return loss
