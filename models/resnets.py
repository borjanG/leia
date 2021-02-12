#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: borjangeshkovski
"""
##------------#
import torch.nn as nn
import torch

class ResidualBlock(nn.Module):
    """
    https://arxiv.org/pdf/1806.10909.pdf
    """
    def __init__(self, data_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim

        # self.mlp = nn.Sequential(
        #     nn.Linear(data_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, data_dim),
        #     nn.ReLU()
        # )

        self.mlp = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim)
        )

    def forward(self, x):
        return x + self.mlp(x)

class ResNet(nn.Module):
    def __init__(self, data_dim, hidden_dim, num_layers, output_dim=1,
                 is_img=False):
        super(ResNet, self).__init__()
        residual_blocks = \
            [ResidualBlock(data_dim, hidden_dim) for _ in range(num_layers)]
        self.residual_blocks = nn.Sequential(*residual_blocks)
        self.linear_layer = nn.Linear(data_dim, output_dim)
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.is_img = is_img

    def forward(self, x, return_features=False):

        traj = list()                       
        traj.append(self.residual_blocks[0](x.view(x.size(0),-1)))      # to store the states/features over layers
        if self.is_img:
            for k in range(1, self.num_layers):
                traj.append(self.residual_blocks[k](traj[k-1]))
            features = self.residual_blocks(x.view(x.size(0), -1))
        else:
            features = self.residual_blocks(x)
        
        pred = self.linear_layer(features)
        _traj = [self.linear_layer(_) for _ in traj]
                
        if return_features:
            return features, pred
        return pred, _traj, traj

    @property
    def hidden_dim(self):
        return self.residual_blocks.hidden_dim

