# -*- coding: utf-8 -*-
import torch
device = torch.device('cpu')
from torch.utils.data import DataLoader
from viz.gifs import trajectory_gif, feature_evolution_gif
from viz.plots import get_feature_history, plt_state_component, plt_norm_components, plt_norm_state
from anode.training import Trainer
from anode.models import ODENet
import pickle

with open('data.txt', 'rb') as fp:
    data_line, test = pickle.load(fp)

dataloader = DataLoader(data_line, batch_size=64, shuffle=True)
dataloader_viz = DataLoader(data_line, batch_size=172, shuffle=True)

for inputs, targets in dataloader_viz:
    break

hidden_dim = 2
data_dim = 2
T = 15.0
num_steps = 15
turnpike = True
dt = T/num_steps

anode = ODENet(device, data_dim, hidden_dim, augment_dim=0, non_linearity='tanh')

if turnpike:
	####optimizer_anode = torch.optim.Adam(anode.parameters(), lr=1e-3, weight_decay=dt*0.01)
    
    #L1
    optimizer_anode = torch.optim.Adam(anode.parameters(), lr=1e-3, weight_decay=0)
    #optimizer_anode = torch.optim.SGD(anode.parameters(), lr=1e-3)
else:
	optimizer_anode = torch.optim.Adam(anode.parameters(), lr=1e-3, weight_decay=dt*0.1)

trainer_anode = Trainer(anode, optimizer_anode, device)
num_epochs = 1000
visualize_features = True
if visualize_features:
    feature_history = get_feature_history(trainer_anode, dataloader, 
                                          inputs, targets, num_epochs)
else:
    trainer_anode.train(dataloader, num_epochs)

trajectory_gif(anode, inputs, targets, timesteps=num_steps)
plt_norm_components(anode, inputs, targets, timesteps=num_steps, save_fig='norm.pdf')
plt_norm_state(anode, inputs, targets, timesteps=num_steps)
feature_evolution_gif(feature_history, targets)
from viz.plots import input_space_plt
input_space_plt(anode, num_steps=1000)

##--------------#
## Saving the weights:
pars = []
for param_tensor in anode.state_dict():
   pars.append(anode.state_dict()[param_tensor])
   #print(param_tensor, "\t", anode.state_dict()[param_tensor])

with open("controls.txt", "wb") as fp:
   pickle.dump(pars, fp)
##--------------#
