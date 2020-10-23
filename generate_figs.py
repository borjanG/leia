# -*- coding: utf-8 -*-

##--------------#
import torch
device = torch.device('cpu')
from torch.utils.data import DataLoader
from viz.gifs import trajectory_gif, feature_evolution_gif
from viz.plots import single_feature_plt, get_feature_history, plt_x_component, plt_y_component, plt_z_component, plt_norm_components
from anode.training import Trainer
from anode.models import ODENet
import pickle
##--------------#

##--------------#
##.. Generate data
with open('data.txt', 'rb') as fp:
    data_line = pickle.load(fp)

dataloader = DataLoader(data_line, batch_size=64, shuffle=True)
##.. Visualize a batch of data
dataloader_viz = DataLoader(data_line, batch_size=128, shuffle=True)
for inputs, targets in dataloader_viz:
    break
##--------------#

##--------------#
## Set up model..
hidden_dim = 2
data_dim = 2	
anode = ODENet(device, data_dim, hidden_dim, augment_dim=0, non_linearity='tanh')

##.. Turnpike (weight_decay=1? for 2d, T=45, num_steps=180)
T = 45.0
num_steps = 180
dt = T/num_steps

##.. No turnpike (weight_decay=dt works well in 2d, dt*0.1 in 3d, for T=45 and num_steps=180.)
#T = 81
#num_steps = T
#dt = 0.1*T/num_steps
#num_steps = int(pow(T, 1.5))
#dt = T/pow(T, 1.5)
##--------------#

#optimizer_anode = torch.optim.Adam(anode.parameters(), lr=1e-3, weight_decay=dt*0.25)
optimizer_anode = torch.optim.Adam(anode.parameters(), lr=1e-3, weight_decay=dt*4)

##--------------#
##.. Set up trainer
trainer_anode = Trainer(anode, optimizer_anode, device)
num_epochs = 200
##.. Optionally record how the features evolve during training
visualize_features = True
if visualize_features:
    feature_history = get_feature_history(trainer_anode, dataloader, 
                                          inputs, targets, num_epochs)
else:
    trainer_anode.train(dataloader, num_epochs)
##--------------#

##--------------#
##.. Viz stage
plt_x_component(anode, inputs, targets, timesteps=num_steps, save_fig='first.pdf')
plt_y_component(anode, inputs, targets, timesteps=num_steps, save_fig='second.pdf')
#plt_z_component(anode, inputs, targets, timesteps=num_steps, save_fig='third.pdf')
trajectory_gif(anode, inputs, targets, timesteps=num_steps)
plt_norm_components(anode, inputs, targets, timesteps=num_steps, save_fig='norm.pdf')
feature_evolution_gif(feature_history, targets)
##.. To see the infered function:
#from viz.plots import input_space_plt
#input_space_plt(anode)

##.. Charlotte's idea:
# import imageio
# import os
# base_filename = "first"
# filename = 'first_gif.gif'
# imgs = []
# for i in range(1,num_steps):
#     img_file = base_filename + "{}.png".format(i)
#     plt_x_component(anode, inputs, targets, timesteps=i, save_fig=img_file)
#     imgs.append(imageio.imread(img_file))
#     os.remove(img_file) 
# imageio.mimwrite(filename, imgs)
##--------------#

##--------------#
##.. If interested in saving the weights:
# pars = []
# for param_tensor in anode.state_dict():
# 	pars.append(anode.state_dict()[param_tensor])
# 	print(param_tensor, "\t", anode.state_dict()[param_tensor])

# with open("text.txt", "wb") as fp:
# 	pickle.dump(pars, fp)
##--------------#
