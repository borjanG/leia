#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: borjangeshkovski
"""
##------------#
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.colors import LinearSegmentedColormap

categorical_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
all_categorical_colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                          '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                          '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                          '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']
T = 10
time_steps = 10
dt = T/time_steps
integration_time = torch.linspace(0., T, time_steps)

def plt_state_component(model, inputs, targets, timesteps, component, save_fig='component.pdf'):
    """
    I forgot..
    """
    from matplotlib import rc
    rc("text", usetex = True)
    font = {'size'   : 18}
    rc('font', **font)
    
    alpha = 0.75

    if hasattr(model, 'num_layers'):
        # A faire.. ResNet (MNIST):
        ends, _, trajectories = model(inputs)
        trajectories = np.asarray(trajectories)
        #color = ..
    else:
        if False in (t < 2 for t in targets): 
            color = ['crimson' if targets[i, 0] == 0.0 else 'dodgerblue' if targets[i,0] == 1.0 else 'green' for i in range(len(targets))]
        else: 
            color = ['crimson' if targets[i, 0] > 0.0 else 'dodgerblue' for i in range(len(targets))]
        trajectories = model.flow.trajectory(inputs, timesteps).detach()
    
    inputs_aug = inputs
    
    for i in range(inputs_aug.shape[0]):
        if hasattr(model, 'num_layers'):
            # ResNet
            y_traj = [x[i][component].detach().numpy() for x in trajectories]
        else: 
            trajectory = trajectories[:, i, :]
            y_traj = trajectory[:, component].numpy()

        ax = plt.gca()
        ax.set_facecolor('whitesmoke')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.title(r'Component of $\mathbf{x}_{i}(t)$', fontsize=12)
        plt.xlabel(r'$t$ (layers)')
        plt.plot(integration_time, y_traj, c=color[i], alpha=alpha, linewidth=0.75)
        ax.set_xlim([0, T])
        plt.rc('grid', linestyle="dotted", color='lightgray')
        ax.grid(True)
    
    if len(save_fig):
        plt.savefig(save_fig, format='pdf', bbox_inches='tight')
        plt.clf()
        plt.close()

def plt_train_error(model, inputs, targets, timesteps, save_fig='train_error.pdf'):
    """
    Plot the training error over each layer / time (not epoch!)
    """
    from matplotlib import rc
    from scipy.interpolate import interp1d
    rc("text", usetex = True)
    font = {'size'   : 13}
    rc('font', **font)
    
    alpha = 0.9
    if hasattr(model, 'num_layers'):
        ## ResNet
        ends, _, traj = model(inputs)
        traj = np.asarray(traj)
        _ = np.asarray(_)                   #traj -> (40, 256, 784)
        #x_norm = [torch.norm(traj[k]) for k in range(timesteps)]
        #x_proj_norm = [torch.norm(_[k]) for k in range(timesteps)]
    else:
        ends, _ = model(inputs)
        _ = _.detach()
    
    loss = nn.CrossEntropyLoss()            # or nn.MSELoss()
    error = [loss(_[k], targets) for k in range(timesteps)]
    
    f2 = interp1d(integration_time, error, kind='cubic', fill_value="extrapolate")
    _time = torch.linspace(0., T, 180)

    ax = plt.gca()
    ax.set_facecolor('whitesmoke')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.title(r'Decay of training error', fontsize=13)
    plt.xlabel(r'$t$ (layers)')
    
    # The training error
    plt.plot(_time, f2(_time), c='tab:red', alpha=alpha, linewidth=2.25, label=r'$\mathcal{E}(\mathbf{x}(t))$')
    ax.legend(prop={'size':10}, loc="upper right", frameon=True)
    ax.set_xlim([0, int(T)])
    plt.rc('grid', linestyle="dotted", color='lightgray')
    ax.grid(True)

    if len(save_fig):
        plt.savefig(save_fig, format='pdf', bbox_inches='tight')
        plt.clf()
        plt.close()  

def plt_norm_state(model, inputs, timesteps, save_fig='norm_state.pdf'):
    """
    Plot the norm of the state trajectory x(t) and the projection Px(t) over time/layer t.
    """
    from matplotlib import rc
    from scipy.interpolate import interp1d
    rc("text", usetex = True)
    font = {'size'   : 13}
    rc('font', **font)
    
    alpha = 0.9    

    if not hasattr(model, 'num_layers'):
        trajectories = model.flow.trajectory(inputs, timesteps).detach()
        ends, _ = model(inputs)
        _ = _.detach()
        x_norm = [np.linalg.norm(trajectories[k, :, :], ord = 'fro') for k in range(timesteps)]
        _norm = [np.linalg.norm(_[k, :], ord = 'fro') for k in range(timesteps)]
        # #x_norm_0 = [np.linalg.norm(trajectories[k, :, 0]) for k in range(timesteps)]
        # #x_norm_1 = [np.linalg.norm(trajectories[k, :, 1]) for k in range(timesteps)]
    else:
        ## ResNet
        ends, _, traj = model(inputs)
        traj = np.asarray(traj)
        _ = np.asarray(_)               #traj -> (40, 256, 784)
        x_norm = [torch.norm(traj[k]) for k in range(timesteps)]
        _norm = [torch.norm(_[k]) for k in range(timesteps)]
    
    f1 = interp1d(integration_time, x_norm, kind='cubic', fill_value="extrapolate")
    f2 = interp1d(integration_time, _norm, kind='cubic', fill_value="extrapolate")
    _time = torch.linspace(0., T, 180)

    ax = plt.gca()
    ax.set_facecolor('whitesmoke')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.title('Stability of norms', fontsize=13)
    plt.xlabel(r'$t$ (layers)')
    plt.plot(_time, f1(_time), c='tab:purple', alpha=alpha, linewidth=2.25, label=r'$|\mathbf{x}(t)|^2$')
    plt.plot(_time, f2(_time), c='tab:orange', alpha=alpha, linewidth=2.25, label=r'$|P\mathbf{x}(t)|^2$')
    ax.legend(prop={'size':10}, loc="upper left", frameon=True)
    ax.set_xlim([0, T])
    plt.rc('grid', linestyle="dotted", color='lightgray')
    ax.grid(True)

    if len(save_fig):
        plt.savefig(save_fig, format='pdf', bbox_inches='tight')
        plt.clf()
        plt.close()  

def plt_norm_control():
    """
    Plot the norm of the control parameters u(t) over time/layer t.
    """
    from matplotlib import rc
    from scipy.interpolate import interp1d
    rc("text", usetex = True)
    font = {'size'   : 13}
    rc('font', **font)
    alpha = 0.9

    import pickle
    dump = []
    with open('plots/controls.txt', 'rb') as fp:
        dump = pickle.load(fp)

    simple = True
    if simple:
        # For now, only works for neural ODEs; sigma inside/outside.
        w_norm = [dump[k].abs().sum() for k in range(0, 2*time_steps, 2)]
        b_norm = [dump[k].abs().sum() for k in range(0, 2*time_steps) if k%2==1]
        ctrl_norm = [x+y for x,y in zip(w_norm, b_norm)]
    else:
        w1_norm = [dump[k].abs().sum() for k in range(0, 4*time_steps) if k%4==0]
        b1_norm = [dump[k].abs().sum() for k in range(0, 4*time_steps) if k%4==1]
        w2_norm = [dump[k].abs().sum() for k in range(0, 4*time_steps) if k%4==2]
        b2_norm = [dump[k].abs().sum() for k in range(0, 4*time_steps) if k%4==3]
        w_norm = [x+y for x,y in zip(w1_norm, w2_norm)]
        b_norm = [x+y for x,y in zip(b1_norm, b2_norm)]
        ctrl_norm = [x+y for x,y in zip(w_norm, b_norm)]

    f1 = interp1d(integration_time, ctrl_norm, kind='next', fill_value="extrapolate")
    _time = torch.linspace(0., T, 150)
    
    ax = plt.gca()
    ax.set_facecolor('whitesmoke')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.title(r'Parameter sparsity: $M=8$', fontsize=13)
    plt.xlabel(r'$t$ (layers)')
    plt.plot(_time, f1(_time), c='tab:blue', alpha=alpha, linewidth=2.25, label=r'$|u(t)|$')
    ax.legend(prop={'size':10}, loc="upper right", frameon=True)
    ax.set_xlim([0, T])
    plt.rc('grid', linestyle="dotted", color='lightgray')
    ax.grid(True)

    save_fig = 'controls.pdf'
    if len(save_fig):
        plt.savefig(save_fig, format='pdf', bbox_inches='tight')
        plt.clf()
        plt.close() 

def feature_plot(feature_history, targets, alpha=0.9, filename='features.pdf'):
    """
    Plots the output features before projection to label space.
    Only works in 2d and 3d.
    """
    
    base_filename = filename[:-4]

    ## We focus on 3 colors at most
    if False in (t < 2 for t in targets): 
        color = ['mediumpurple' if targets[i] == 2.0 else 'gold' if targets[i] == 0.0 else 'mediumseagreen' for i in range(len(targets))]
    else:
        #color = ['crimson' if targets[i, 0] > 0.0 else 'dodgerblue' for i in range(len(targets))]
        color = ['crimson' if targets[i] > 0.0 else 'dodgerblue' for i in range(len(targets))]
    
    num_dims = feature_history[0].shape[1]
    features = feature_history[-1]
    i = len(feature_history)
    
    if num_dims == 2:
        ax = plt.gca()
        ax.set_facecolor('whitesmoke')                      #Gray background
        ax.set_axisbelow(True)                              #Axis beneath data
        ax.xaxis.grid(color='lightgray', linestyle='dotted')
        ax.yaxis.grid(color='lightgray', linestyle='dotted')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        label_size = 12
        plt.rcParams['xtick.labelsize'] = label_size
        plt.rcParams['ytick.labelsize'] = label_size 
        plt.xlabel(r'$x_1$', fontsize=13)
        plt.ylabel(r'$x_2$', fontsize=13)
        plt.scatter(features[:, 0].numpy(), features[:, 1].numpy(), c=color,
                    alpha=alpha, marker = 'o', linewidth=0.65, edgecolors='black')
    elif num_dims == 3:
        fig = plt.figure()
        ax = Axes3D(fig)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        label_size = 12
        plt.rcParams['xtick.labelsize'] = label_size
        plt.rcParams['ytick.labelsize'] = label_size
        ax.scatter(features[:, 0].numpy(), features[:, 1].numpy(), features[:, 2].numpy(),
                   c=color, alpha=alpha, marker = 'o', linewidth=0.65, edgecolors='black')
        plt.rc('grid', linestyle="dotted", color='lightgray')
        ax.grid(True)
        plt.locator_params(nbins=4)
        
    plt.savefig(base_filename + "{}.pdf".format(i), format='pdf', bbox_inches='tight')
    plt.clf()
    plt.close()

def plt_classifier(model, plot_range=(-2.0, 2.0), num_steps=201, save_fig='generalization.pdf'):
    """
    Plots the final classifier; train and test data are superposed.
    Only for toy cloud data.
    """
    import matplotlib as mpl
    from matplotlib import rc
    import seaborn as sns
    from torch.utils.data import DataLoader
    import pickle
    rc("text", usetex = True)
    font = {'size'   : 13}
    rc('font', **font)
    cross_entropy = True

    with open('data.txt', 'rb') as fp:
        data_line, test = pickle.load(fp)
    
    dataloader_viz = DataLoader(data_line, batch_size=800, shuffle=True)
    test_viz = DataLoader(test, batch_size = 80, shuffle=True)
    for inputs, targets in dataloader_viz:
        break    
    for test_inputs, test_targets in test_viz:
        break
    
    if False in (t < 2 for t in targets): 
        color = ['mediumpurple' if targets[i] == 2.0 else 'gold' if targets[i] == 0.0 else 'mediumseagreen' for i in range(len(targets))]
        test_color = ['mediumpurple' if test_targets[i] == 2.0 else 'gold' if test_targets[i] == 0.0 else 'mediumseagreen' for i in range(len(test_targets))]
        cmap = mpl.cm.get_cmap("viridis_r")
        bounds = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    else:
        color = ['crimson' if targets[i] > 0.0 else 'dodgerblue' for i in range(len(targets))]
        test_color = ['crimson' if test_targets[i] > 0.0 else 'dodgerblue' for i in range(len(test_targets))]
        cmap = sns.diverging_palette(250, 10, s=50, l=30, n=9, center="light", as_cmap=True)
        if cross_entropy:
            bounds = [0.0, 0.1, 0.25, 0.35, 0.5, 0.65, 0.75, 0.9, 1.0]          # cross-entropy labels
        else: 
            bounds = [-1.0, -0.75,-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]       # mse labels
    
    grid = torch.zeros((num_steps * num_steps, 2))
    idx = 0
    for x1 in np.linspace(plot_range[0], plot_range[1], num_steps):
        for x2 in np.linspace(plot_range[0], plot_range[1], num_steps):
            grid[idx, :] = torch.Tensor([x1, x2])
            idx += 1

    if not cross_entropy:
        predictions, traj = model(grid)
        vmin, vmax = -1.05, 1.05
    else:
        pre_, traj = model(grid)
        m = nn.Softmax()
        predictions = m(pre_)
        predictions = torch.argmax(predictions, 1)
        vmin = 0.0
        vmax = 2.05 if False in (t < 2 for t in targets) else 1.05 
    
    pred_grid = predictions.view(num_steps, num_steps).detach()
    
    _x = np.linspace(plot_range[0], plot_range[1], num_steps)
    _y = np.linspace(plot_range[0], plot_range[1], num_steps)
        
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    fig = plt.figure()

    X_new, Y_new = np.meshgrid(_x,_y)
    i = plt.contourf(X_new, Y_new, pred_grid, vmin=vmin, vmax=vmax, cmap=cmap, norm=norm, alpha=1)

    cb = fig.colorbar(i)
    cb.ax.tick_params(size=0)
    plt.tick_params(axis='both', which='both', bottom=False, top=False,
                        labelbottom=False, right=False, left=False,
                        labelleft=False)
    
    plt.scatter(inputs[:,0], inputs[:,1], c=color, alpha=0.95, marker = 'o', linewidth=0.45, edgecolors='black', label='train')
    plt.scatter(test_inputs[:,0], test_inputs[:, 1], c=test_color, alpha=0.95, marker='o', linewidth=1.75, edgecolors='black', label='test')
    
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', mew=0.45, mec='black', label='train',
                          markerfacecolor='lightgray', markersize=7),
                        Line2D([0], [0], marker='o', color='w', mew=1.75, mec='black', label='test',
                          markerfacecolor='lightgray', markersize=7)]

    plt.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(-0.315,1.025), frameon=False)
    plt.title('Generalization outside training data', fontsize=13)
    plt.xlabel(r'$x_1$', fontsize=13)
    plt.ylabel(r'$x_2$', fontsize=13)

    if len(save_fig):
        plt.savefig(save_fig, format='pdf', bbox_inches='tight')
        plt.clf()
        plt.close()

def get_feature_history(trainer, dataloader, inputs, targets, num_epochs):
    
    feature_history = []
    features, _ = trainer.model(inputs, return_features=True)
    feature_history.append(features.detach())

    for i in range(num_epochs):
        trainer.train(dataloader, 1)
        features, _ = trainer.model(inputs, return_features=True)
        feature_history.append(features.detach())
    return feature_history
