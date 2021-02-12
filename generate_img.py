#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: borjangeshkovski
"""
##------------#
import torch
device = torch.device('cpu')
from models.resnets import ResNet
from models.training import Trainer
from data.dataloaders import mnist
from plots.plots import plt_train_error, plt_norm_state, plt_norm_control
from plots.gifs import mnist_gif
import pickle

##------------#
## Data:  
data_loader, test_loader = mnist(256)   #img_size = (1, 28, 28)
for inputs, targets in data_loader:
    break

##--------------#
## Setup:
output_dim = 10
num_epochs = 20
num_layers = 10
hidden_dim = 32

import time
start_time = time.time()
model = ResNet(pow(28,2), hidden_dim=hidden_dim, num_layers=num_layers, output_dim=10, is_img=True)

ell1 = False
weight_decay = 0 if ell1 else 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
trainer = Trainer(model, optimizer, device)
trainer.train(data_loader, num_epochs)
print("--- %s seconds ---" % (time.time() - start_time))

##--------------#
## Plots:
component = 3
#plt_state_component(model, inputs.view(inputs.size(0),-1), targets, timesteps=num_layers, component=component, save_fig='{}.pdf'.format(compoent))
plt_train_error(model, inputs.view(inputs.size(0),-1), targets, num_layers, save_fig='train_error.pdf')
plt_norm_state(model, inputs.view(inputs.size(0),-1), num_layers, save_fig='norm_state.pdf')
mnist_gif(model, inputs.view(inputs.size(0),-1), num_layers, component)
##--------------#
## Saving the weights:
pars = []
for param_tensor in model.state_dict():
   pars.append(model.state_dict()[param_tensor])
   #print(param_tensor, "\t", anode.state_dict()[param_tensor])

with open("plots/controls.txt", "wb") as fp:
   pickle.dump(pars, fp)
plt_norm_control()