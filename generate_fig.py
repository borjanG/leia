#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: borjangeshkovski
"""
##------------#
import torch
device = torch.device('cpu')
from torch.utils.data import DataLoader
from plots.gifs import trajectory_gif
from plots.plots import get_feature_history, plt_train_error, plt_norm_state, plt_norm_control, plt_classifier, feature_plot
from models.training import Trainer
from models.neural_odes import NeuralODE
import pickle

##--------------#
## Data: 
with open('data.txt', 'rb') as fp:
    data_line, test = pickle.load(fp)
dataloader = DataLoader(data_line, batch_size=64, shuffle=True)
dataloader_viz = DataLoader(data_line, batch_size=172, shuffle=True)
for inputs, targets in dataloader_viz:
    break

##--------------#
## Setup:
hidden_dim, data_dim = 2, 2
T, num_steps = 10.0, 15
dt = T/num_steps
turnpike, ell1 = True, True

if turnpike:
    weight_decay = 0 if ell1 else dt*0.01
else: 
    weight_decay = dt*0.1

anode = NeuralODE(device, data_dim, hidden_dim, augment_dim=0, non_linearity='relu')
optimizer_anode = torch.optim.Adam(anode.parameters(), lr=1e-3, weight_decay=weight_decay)
trainer_anode = Trainer(anode, optimizer_anode, device)
num_epochs = 1000
visualize_features = True

import time
start_time = time.time()
if visualize_features:
    feature_history = get_feature_history(trainer_anode, dataloader, 
                                          inputs, targets, num_epochs)
else:
    trainer_anode.train(dataloader, num_epochs)
print("--- %s seconds ---" % (time.time() - start_time))

##--------------#
## Plots:
plt_norm_state(anode, inputs, timesteps=num_steps)
plt_train_error(anode, inputs, targets, timesteps=num_steps)
feature_plot(feature_history, targets)
plt_classifier(anode, num_steps=1000)
#trajectory_gif(anode, inputs, targets, timesteps=num_steps)
##--------------#
## Saving the weights:
pars = []
for param_tensor in anode.state_dict():
   pars.append(anode.state_dict()[param_tensor])
   #print(param_tensor, "\t", anode.state_dict()[param_tensor])

with open("plots/controls.txt", "wb") as fp:
   pickle.dump(pars, fp)
plt_norm_control()
##--------------#
