# -*- coding: utf-8 -*-

import torch
device = torch.device('cpu')
from torch.utils.data import DataLoader
from viz.gifs import trajectory_gif, feature_evolution_gif
from viz.plots import single_feature_plt, get_feature_history, plt_x_component, plt_y_component
from anode.training import Trainer
from anode.models import ODENet
import pickle

with open('data.txt', 'rb') as fp:
    data_line = pickle.load(fp)

dataloader = DataLoader(data_line, batch_size=64, shuffle=True)

##.. Visualize a batch of data
dataloader_viz = DataLoader(data_line, batch_size=128, shuffle=True)

for inputs, targets in dataloader_viz:
    break

#single_feature_plt(inputs, targets, 'sines.pdf')

hidden_dim = 2
data_dim = 1		
anode = ODENet(device, data_dim, hidden_dim, augment_dim=1, non_linearity='tanh')


T = 20.0
#dt = 0.4

#T = 81
#num_steps = int(pow(T, 1.5))
#dt = T/pow(T, 1.5)

num_steps = 50
#num_steps = 13
dt = T/num_steps

#optimizer_anode = torch.optim.Adam(anode.parameters(), lr=1e-3, weight_decay=dt*0.5)
optimizer_anode = torch.optim.Adam(anode.parameters(), lr=1e-3, weight_decay=dt)
##optimizer_anode = torch.optim.SGD(anode.parameters(), lr=1e-3, momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=True)

##.. Set up trainer
trainer_anode = Trainer(anode, optimizer_anode, device)
num_epochs = 100

##.. Optionally record how the features evolve during training
visualize_features = True

if visualize_features:
    feature_history = get_feature_history(trainer_anode, dataloader, 
                                          inputs, targets, num_epochs)
else:
    trainer_anode.train(dataloader, num_epochs)

#plt_x_component(anode, inputs, targets, timesteps=num_steps, save_fig='first.pdf')
#plt_y_component(anode, inputs, targets, timesteps=num_steps, save_fig='second.pdf')
trajectory_gif(anode, inputs, targets, timesteps=num_steps)


from viz.plots import input_space_plt

#input_space_plt(anode)

feature_evolution_gif(feature_history, targets)

#from viz.plots import input_space_plt

#input_space_plt(anode)

# ##.. If interested in saving the weights:
# pars = []
# for param_tensor in anode.state_dict():
#    pars.append(anode.state_dict()[param_tensor])
#    #print(param_tensor, "\t", anode.state_dict()[param_tensor])

# with open("text.txt", "wb") as fp:
#   pickle.dump(pars, fp)
