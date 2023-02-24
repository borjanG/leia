#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: borjangeshkovski
"""

import torch
device = torch.device('cpu')
from models.resnets import ResNet
from models.training import Trainer
from data.dataloaders import mnist, cifar10, fashion_mnist
from plots.plots import plt_train_error, plt_norm_state, plt_norm_control
from plots.gifs import mnist_gif, cifar_gif
import pickle

style = 'mnist'
fashion = False
if style == 'mnist':
    if fashion: 
        data_loader, test_loader = fashion_mnist(256)    
    else: 
        data_loader, test_loader = mnist(256)               #img_size = (1, 28, 28)
    pixel, channel = 28, 1
else:
    data_loader, test_loader = cifar10()
    pixel, channel = 32, 3
for inputs, targets in data_loader:
    break

output_dim = 10
num_epochs = 15
num_layers = 20
#hidden_dim = pixel+4
hidden_dim = 32              #4-8 for mnist is ok.

import time
start_time = time.time()
model = ResNet(channel*pow(pixel, 2), 
                hidden_dim=hidden_dim, 
                num_layers=num_layers, 
                output_dim=10, 
                is_img=True)

bound = 250.
weight_decay = 0 if bound>0. else 0.01
optimizer = torch.optim.Adam(model.parameters(), 
                                lr=1e-3, 
                                weight_decay=weight_decay)
trainer = Trainer(model, 
                    optimizer, 
                    device, 
                    turnpike=True, 
                    bound=bound)
trainer.train(data_loader, num_epochs)
print("--- %s seconds ---" % (time.time() - start_time))

component = 10
plt_train_error(model, 
                inputs.view(inputs.size(0),-1), 
                targets, 
                num_layers, 
                save_fig='train_error.pdf')

plt_norm_state(model, 
                inputs.view(inputs.size(0),-1), 
                num_layers, 
                save_fig='norm_state.pdf')

if style == 'mnist':
    mnist_gif(model, 
                inputs.view(inputs.size(0),-1), 
                num_layers, 
                component)
else: 
    cifar_gif(model, 
                inputs.view(inputs.size(0),-1), 
                num_layers, 
                component)

pars = []
for param_tensor in model.state_dict():
    pars.append(model.state_dict()[param_tensor])
    #print(param_tensor, "\t", anode.state_dict()[param_tensor])

with open("plots/controls.txt", "wb") as fp:
    pickle.dump(pars, fp)
plt_norm_control()