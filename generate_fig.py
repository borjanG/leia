#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: borjangeshkovski 
To do: replace by a Jupyter Notebook.
"""

import torch
device = torch.device('cpu')
from torch.utils.data import DataLoader
from plots.gifs import trajectory_gif
from plots.plots import get_feature_history, plt_train_error, plt_norm_state 
from plots.plots import plt_classifier, feature_plot
from models.training import Trainer
from models.neural_odes import NeuralODE
import pickle

# =============================================================================
# DATA
# =============================================================================

with open('data.txt', 'rb') as fp:
    data_line, test = pickle.load(fp)
dataloader = DataLoader(data_line, 
                        batch_size=150, 
                        shuffle=True)

dataloader_viz = DataLoader(data_line, 
                            batch_size=150, 
                            shuffle=True)

for inputs, targets in dataloader_viz:
    break

# =============================================================================
# SETUP
# =============================================================================

hidden_dim, data_dim = 50, 2
T, num_steps = 5.0, 16
dt = T/num_steps
turnpike, cross_entropy, L1 = True, False, False

if L1:
    # Use Euler scheme for L^1+L^infty, seems to work better.
    bound = 8.0
    weight_decay = 0.0
else:
    # L^2: Midpoint scheme works good too.
    bound = 0.0
    weight_decay = 0.01*dt

# =============================================================================
# MODEL
# =============================================================================

anode = NeuralODE(device, 
                  data_dim, 
                  hidden_dim, 
                  augment_dim=1, 
                  non_linearity='relu',
                  architecture='bottleneck', 
                  T=T, 
                  time_steps=num_steps, 
                  fixed_projector=False, 
                  cross_entropy=cross_entropy)

# =============================================================================
# OPTIMIZER
# =============================================================================

optimizer_anode = torch.optim.Adam(anode.parameters(), 
                                   lr=1e-3, 
                                   weight_decay=weight_decay)

trainer_anode = Trainer(anode, 
                        optimizer_anode, 
                        device, 
                        cross_entropy=cross_entropy, 
                        turnpike=turnpike,
                        bound=bound, 
                        fixed_projector=fp)
num_epochs = 2500
visualize_features = True

import time
start_time = time.time()
if visualize_features:
    feature_history = get_feature_history(trainer_anode, 
                                          dataloader, 
                                          inputs, 
                                          targets, 
                                          num_epochs)
else:
    trainer_anode.train(dataloader, num_epochs)

# =============================================================================
# SAVING WEIGHTS
# =============================================================================
    
pars = []
for param_tensor in anode.state_dict():
    pars.append(anode.state_dict()[param_tensor])
    #print(param_tensor, "\t", anode.state_dict()[param_tensor])

with open("plots/controls.txt", "wb") as fp:
    pickle.dump(pars, fp)

# =============================================================================
# PLOTS
# =============================================================================

#plt_norm_state(anode, inputs, timesteps=num_steps)
#plt_train_error(anode, inputs, targets, timesteps=num_steps)
#plt_norm_control(anode)
#feature_plot(feature_history, targets)
#trajectory_gif(anode, inputs, targets, timesteps=num_steps)

with open('data.txt', 'rb') as fp:
    data_line, test = pickle.load(fp)

dataloader_viz = DataLoader(data_line, batch_size=800, shuffle=True)
test_viz = DataLoader(test, batch_size = 80, shuffle=True)

for inputs, targets in dataloader_viz:
    break    
for test_inputs, test_targets in test_viz:
    break

for t in range(0, num_steps):
    plt_classifier(anode, 
                    inputs, 
                    test_inputs, 
                    targets, 
                    test_targets, 
                    t=t, 
                    num_steps=150)

plt_classifier(anode, 
               inputs, 
               test_inputs, 
               targets, 
               test_targets, 
               t=-1, 
               num_steps=150)

print("--- Runtime: %s seconds ---" % (time.time() - start_time))

# Plotting matrix weights
# =============================================================================
# # import matplotlib.pyplot as plt
# # import numpy as np
# # for k in range(len(pars)):
# #     if k%2==0:
# #     #if 1>0:
# #         #plt.imshow((abs(pars[k])>=5*1e-4)*pars[k], cmap="binary")
#         
# #         pars[k][abs(pars[k])>=1e-2] = 1
# #         pars[k][abs(pars[k])<1e-2] = 0
# #         plt.pcolormesh(pars[k], edgecolors='lightgray', linewidth=1, cmap='binary')
#         
# #         #plt.pcolormesh((abs(pars[k])>1e-3)*pars[k], edgecolors='lightgray', linewidth=0.25, cmap='binary')
# 
# #         #plt.matshow(abs(pars[k]), cmap="binary")
# #         #plt.colorbar()
#         
# #         # ax = plt.gca();
# 
# #         # #  Major ticks
# #         # ax.set_xticks(np.arange(0, 3, 1))
# #         # ax.set_yticks(np.arange(0, 3, 1))
# 
#         
# #         # # Minor ticks
# #         # ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
# #         # ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
# 
# #         # # Gridlines based on minor ticks
# #         # ax.grid(which='minor', color='b', linestyle='-', linewidth=2)
#         
# #         plt.gca().xaxis.set_ticklabels([])
# #         plt.gca().yaxis.set_ticklabels([])
# #         plt.clim(0, 1)
# #         ax = plt.gca()
# #         ax.set_aspect('equal')
# #         #plt.axis('off')
# #         plt.savefig("mat_{}.pdf".format(k), format='pdf', bbox_inches='tight')
# #         plt.clf()
# #         plt.close() 
# =============================================================================
