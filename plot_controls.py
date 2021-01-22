import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.colors import LinearSegmentedColormap

T = 15.0
time_steps = 15
dt = T/time_steps
integration_time = torch.linspace(0., T, time_steps)

dump = []
with open('controls.txt', 'rb') as fp:
	dump = pickle.load(fp)

from matplotlib import rc
from scipy.interpolate import interp1d
rc("text", usetex = True)
font = {'size'   : 13}
rc('font', **font)
alpha = 0.9

w_norm = [dump[k].abs().sum() for k in range(0, 2*time_steps, 2)]
b_norm = [dump[k].abs().sum() for k in range(0, 2*time_steps) if k%2==1]

ctrl_norm = [x+y for x,y in zip(w_norm, b_norm)]
#ctrl_norm[0] = torch.tensor(5.0)
#ctrl_norm[1] = torch.tensor(5.0)

#f1 = interp1d(integration_time, ctrl_norm, kind='linear', fill_value="extrapolate")
#_time = torch.linspace(0., T, 180)

ax = plt.gca()
ax.set_facecolor('whitesmoke')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.title(r'Parameter sparsity: $M=5$', fontsize=13)
plt.xlabel(r'$t$ (layers)')
#plt.plot(_time, f1(_time), c='tab:blue', alpha=alpha, linewidth=2.25, label=r'$|u(t)|$')
plt.plot(integration_time, ctrl_norm, c='tab:blue', alpha=alpha, linewidth=2.25, label=r'$|u(t)|_1$')

ax.legend(prop={'size':10}, loc="upper right", frameon=True)
ax.set_xlim([0, T])
plt.rc('grid', linestyle="dotted", color='lightgray')
ax.grid(True)

save_fig = 'controls.pdf'

if len(save_fig):
    plt.savefig(save_fig, format='pdf', bbox_inches='tight')
    plt.clf()
    plt.close() 