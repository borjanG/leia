#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: borjangeshkovski
"""

#------------#
import torch
device = torch.device('cpu')
from anode.discrete_models import ResNet
from anode.training import Trainer
from experiments.dataloaders import mnist
from viz.plots import plt_state_component, plt_norm_state
from viz.gifs import mnist_gif
#------------#
 
data_loader, test_loader = mnist(256)
#img_size = (1, 28, 28)

#--------------#
output_dim = 10
num_epochs = 10
num_layers = 40
hidden_dim = 32
#--------------#
 
model = ResNet(pow(28,2), hidden_dim=hidden_dim, num_layers=num_layers, output_dim=10, is_img=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01)
trainer = Trainer(model, optimizer, device, classification=True)
trainer.train(data_loader, num_epochs)

for inputs, targets in data_loader:
    break

#--------#
component = 397
plt_state_component(model, inputs.view(inputs.size(0),-1), targets, timesteps=num_layers, component=component, save_fig='{}.pdf'.format(compoent))
plt_norm_state(model, inputs.view(inputs.size(0),-1), targets, timesteps=num_layers, save_fig='norm.pdf')
mnist_gif(model, inputs.view(inputs.size(0),-1), targets, timesteps=num_layers, component)
#--------#