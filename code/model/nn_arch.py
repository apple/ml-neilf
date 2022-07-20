#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
import numpy as np
import torch
import torch.nn as nn
from model.embedder import get_embedder

class SineLayer(nn.Module):
    ''' Siren layer '''
    
    def __init__(self, 
                 in_features, 
                 out_features, 
                 bias=True, 
                 is_first=False, 
                 omega_0=30, 
                 weight_norm=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

        if weight_norm:
            self.linear = nn.utils.weight_norm(self.linear)

    def init_weights(self):
        if self.is_first:
            nn.init.uniform_(self.linear.weight, 
                             -1 / self.in_features * self.omega_0, 
                             1 / self.in_features * self.omega_0)
        else:
            nn.init.uniform_(self.linear.weight, 
                             -np.sqrt(3 / self.in_features), 
                             np.sqrt(3 / self.in_features))
        nn.init.zeros_(self.linear.bias)

    def forward(self, input):
        return torch.sin(self.linear(input))

class BRDFMLP(nn.Module):

    def __init__(
            self,
            in_dims,
            out_dims,
            dims,
            skip_connection=(),
            weight_norm=True,
            multires_view=0
    ):
        super().__init__()
        self.init_range = np.sqrt(3 / dims[0])

        dims = [in_dims] + dims + [out_dims]
        first_omega = 30
        hidden_omega = 30

        self.embedview_fn = lambda x : x
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view, in_dims)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - in_dims)

        self.num_layers = len(dims)
        self.skip_connection = skip_connection

        for l in range(0, self.num_layers - 1):

            if l + 1 in self.skip_connection:
                out_dim = dims[l + 1] - dims[0] 
            else:
                out_dim = dims[l + 1]

            is_first = (l == 0) and (multires_view == 0)
            is_last = (l == (self.num_layers - 2))
            
            if not is_last:
                omega_0 = first_omega if is_first else hidden_omega
                lin = SineLayer(dims[l], out_dim, True, is_first, omega_0, weight_norm)
            else:
                lin = nn.Linear(dims[l], out_dim)
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
                if weight_norm:
                    lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

            self.last_active_fun = nn.Tanh()

    def forward(self, points):
        init_x = self.embedview_fn(points)
        x = init_x

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_connection:
                x = torch.cat([x, init_x], 1)
                
            x = lin(x)

        x = self.last_active_fun(x)

        return x


class NeILFMLP(nn.Module):

    def __init__(
            self,
            in_dims,
            out_dims,
            dims,
            skip_connection=(),
            position_insertion=(),
            weight_norm=True,
            multires_view=0
    ):
        super().__init__()
        self.init_range = np.sqrt(3 / dims[0])


        d_pos = 3
        d_dir = 3
        dims = dims + [out_dims]

        first_omega = 30
        hidden_omega = 30

        self.embedview_fn = lambda x : x
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view, d_dir)
            self.embedview_fn = embedview_fn
            dims = [input_ch] + dims
        else:
            dims = [d_dir] + dims

        self.num_layers = len(dims)
        self.skip_connection = skip_connection
        self.position_insertion = position_insertion

        for l in range(0, self.num_layers - 1):

            out_dim = dims[l + 1]
            if l + 1 in self.skip_connection:
                out_dim = out_dim - dims[0]
            if l + 1 in self.position_insertion:
                out_dim = out_dim - d_pos

            is_first = (l == 0) and (multires_view == 0)
            is_last = (l == (self.num_layers - 2))
            
            if not is_last:
                omega_0 = first_omega if is_first else hidden_omega
                lin = SineLayer(dims[l], out_dim, True, is_first, omega_0, weight_norm)
            else:
                lin = nn.Linear(dims[l], out_dim)
                nn.init.zeros_(lin.weight)
                nn.init.constant_(lin.bias, np.log(1.5))
                if weight_norm:
                    lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

            self.last_active_fun = torch.exp

    def forward(self, points):

        pose_embed = points[:, 0:3]
        view_embed = self.embedview_fn(points[:, 3:6])
        x = view_embed

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_connection:
                x = torch.cat([x, view_embed], 1)

            if l in self.position_insertion:
                x = torch.cat([x, pose_embed], 1)
                
            x = lin(x)

        x = self.last_active_fun(x)

        return x