# -*- coding: utf-8 -*-

##--------------#
import torch
device = torch.device('cpu')
from torch.utils.data import DataLoader
from viz.gifs import trajectory_gif, feature_evolution_gif
from viz.plots import get_feature_history, plt_state_component, plt_norm_components, plt_norm_state
from anode.training import Trainer
from anode.models import ODENet
import pickle
##--------------#

##--------------#
with open('data.txt', 'rb') as fp:
    data_line = pickle.load(fp)

dataloader = DataLoader(data_line, batch_size=64, shuffle=True)
dataloader_viz = DataLoader(data_line, batch_size=128, shuffle=True)
for inputs, targets in dataloader_viz:
    break
##--------------#

##--------------#
hidden_dim = 2
data_dim = 2	
anode = ODENet(device, data_dim, hidden_dim, augment_dim=0, non_linearity='tanh')

##.. Turnpike (weight_decay=1 for 2d, T=45, num_steps=180)
##.. et meme weight_decay=1, T=20, num_steps=20 avec 500 epochs!
T = 15
num_steps = 30
dt = T/num_steps

##.. No turnpike (weight_decay=dt works well in 2d, dt*0.1 in 3d, for T=45 and num_steps=180.)
#T = 81
#num_steps = T
#dt = 0.1*T/num_steps
#num_steps = int(pow(T, 1.5))
#dt = T/pow(T, 1.5)
##--------------#

optimizer_anode = torch.optim.Adam(anode.parameters(), lr=1e-3, weight_decay=dt)

##--------------#
trainer_anode = Trainer(anode, optimizer_anode, device)
num_epochs = 200
visualize_features = True
if visualize_features:
    feature_history = get_feature_history(trainer_anode, dataloader, 
                                          inputs, targets, num_epochs)
else:
    trainer_anode.train(dataloader, num_epochs)
##--------------#

##--------------#
#plt_state_component(anode, inputs, targets, timesteps=num_steps, component=0, save_fig='first.pdf')
#plt_state_component(anode, inputs, targets, timesteps=num_steps, component=1, save_fig='second.pdf')
trajectory_gif(anode, inputs, targets, timesteps=num_steps)
plt_norm_components(anode, inputs, targets, timesteps=num_steps, save_fig='norm.pdf')
plt_norm_state(anode, inputs, targets, timesteps=num_steps)
feature_evolution_gif(feature_history, targets)

from viz.plots import input_space_plt
input_space_plt(anode)
##--------------#

##--------------#
## Saving the weights:
# pars = []
# for param_tensor in anode.state_dict():
# 	pars.append(anode.state_dict()[param_tensor])
# 	print(param_tensor, "\t", anode.state_dict()[param_tensor])

# with open("text.txt", "wb") as fp:
# 	pickle.dump(pars, fp)
##--------------#
