#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: borjangeshkovski
"""

import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.interpolate import interp1d
import numpy as np
import torch
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D

def plt_train_error(model, 
                    inputs, 
                    targets, 
                    timesteps, 
                    save_fig='train_error.pdf'):
    
    rc("text", usetex = True)
    font = {'size'   : 13}
    rc('font', **font)

    if hasattr(model, 'num_layers'):              
        # ResNet
        ends, _, traj = model(inputs)
        traj = np.asarray(traj)
        _ = np.asarray(_)                                               
        T = model.num_layers
    else:
        ends, _ = model(inputs)
        _ = _.detach()
        ends = ends.detach()
        T = model.T
        
    integration_time = torch.linspace(0., T, timesteps)
    
    if model.cross_entropy:
        loss = nn.CrossEntropyLoss()  
        #loss = nn.MultiMarginLoss()                               
        error = [loss(_[k], targets) for k in range(timesteps)]
    else:
        loss = nn.MSELoss()
        error = [loss(_[k], targets) for k in range(timesteps)]

    # Interpolate to increase smoothness
    f2 = interp1d(integration_time, 
                  error, 
                  kind='linear', 
                  fill_value="extrapolate")
    _time = torch.linspace(0., T, 1*timesteps)

    ax = plt.gca()
    ax.set_axisbelow(True)                              #Axis beneath data
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.xlabel(r'$t_k$ ($k$ is a layer)')
    #plt.ylabel('error')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # The training error
    ax.scatter(torch.linspace(0., T, len(error)), 
               error, 
               s=15,
               c='#BF2633', 
               alpha=1, 
               marker='o',
               edgecolors='black',
               linewidths=0.5, 
               zorder=2)
        
    plt.plot(_time, f2(_time), 
             c='#BF2633', 
             alpha=0.9, 
             linewidth=2.5, 
             label="empirical risk",
             zorder=1)

    ax.xaxis.grid(color='lightgray', linestyle=(0, (1, 10)))
    ax.yaxis.grid(color='lightgray', linestyle=(0, (1, 10)))
    ax.set_xlim([0, int(T)])
    
    labels = ax.get_yticks().tolist()
    labels[-2] = round(float(error[0]), 2)
    for i in range(2, len(labels)-2):
        labels[i] = None
        
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=6)

    ax.set_yticklabels(labels)

    if len(save_fig):
        plt.savefig(save_fig, format='pdf', bbox_inches='tight')
        plt.clf()
        plt.close()  

def plt_norm_state(model, inputs, timesteps, save_fig='norm_state.pdf'):
    rc("text", usetex = True)
    font = {'size'   : 13}
    rc('font', **font)
    alpha = 0.9    

    if not hasattr(model, 'num_layers'):
        trajectories = model.flow.trajectory(inputs, timesteps).detach()
        ends, _ = model(inputs)
        x_norm = [np.linalg.norm(trajectories[k, :, :], ord = 'fro') 
                  for k in range(timesteps)]
        _ = _.detach()
        T = model.T
        if model.cross_entropy:
            _norm = [np.linalg.norm(_[k, :], ord = 'fro') 
                     for k in range(timesteps)]
        else:
            non_linearity = nn.Tanh()
            import pickle
            with open('plots/controls.txt', 'rb') as fp:
                projector = pickle.load(fp)
            _norm = [np.linalg.norm(non_linearity(trajectories[k,:, :].
                                                  matmul(projector[-2].t())
                                                  +projector[-1]), ord='fro') 
                    for k in range(timesteps)]
    else:                                                                       
        # ResNet
        ends, _, traj = model(inputs)
        traj = np.asarray(traj)
        _ = np.asarray(_)                                                       
        x_norm = [torch.norm(traj[k]) for k in range(timesteps)]
        _norm = [torch.norm(_[k]) for k in range(timesteps)]
        T = model.num_layers
    
    integration_time = torch.linspace(0., T, timesteps)
    
    # Interpolate to increase smoothness
    f1 = interp1d(integration_time, 
                  x_norm, 
                  kind='linear', 
                  fill_value="interpolate")
    f2 = interp1d(integration_time, 
                  _norm, kind='linear', 
                  fill_value="interpolate")
    
    _time = torch.linspace(0., T, 10*timesteps)

    ax = plt.gca()
    ax.set_axisbelow(True)                              #Axis beneath data
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.xlabel(r'$t_k$ ($k$ is a layer)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.grid(color='lightgray', linestyle=(0, (1, 10)))
    ax.yaxis.grid(color='lightgray', linestyle=(0, (1, 10)))
    plt.plot(_time, 
             f1(_time), 
             c='#D95319', 
             alpha=alpha, 
             linewidth=2.5, 
             label=r'$|\mathbf{x}(t)|^2$',
             zorder=1)
    plt.plot(_time, 
             f2(_time), 
             c='#EDB120', 
             alpha=0.9, 
             linewidth=2.5,
             zorder=1)
    
    if not hasattr(model, 'num_layers'):
        ax.scatter(torch.linspace(0., T, len(x_norm)), 
                   x_norm, 
                   c='#D95319', 
                   alpha=1, 
                   marker='D',
                   edgecolors='black',
                   linewidths=0.5, 
                   zorder=2)
        ax.scatter(torch.linspace(0., T, len(_norm)), 
                   _norm, 
                   c='#EDB120', 
                   alpha=1, 
                   marker='o',
                   edgecolors='black',
                   linewidths=0.5, 
                   zorder=2)
    
    labels = ax.get_yticks().tolist()
    labels[-2] = round(float(x_norm[-2]))
    for i in range(2, len(labels)-2):
        labels[i] = None
        
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=6)
    ax.set_yticklabels(labels)
    ax.set_xlim([0, T])

    if len(save_fig):
        plt.savefig(save_fig, format='pdf', bbox_inches='tight')
        plt.clf()
        plt.close()  

# def plt_norm_control(model):
#     """
#     Plot the norm of the control parameters u(t) over time/layer t.
#     """
#     rc("text", usetex = True)
#     font = {'size'   : 13}
#     rc('font', **font)

#     import pickle
#     dump = []
#     with open('plots/controls.txt', 'rb') as fp:
#         dump = pickle.load(fp)

#     simple = False
#     T, time_steps = model.T, model.time_steps
#     integration_time = torch.linspace(0., T, time_steps)
    
#     if simple:
#         # For now, L^1 only works for neural ODEs sigma inside/outside.
#         w_norm = [dump[k].abs().sum() 
#                   for k in range(0, 2*time_steps, 2)]
#         b_norm = [dump[k].abs().sum() 
#                   for k in range(0, 2*time_steps) if k%2==1]
#         ctrl_norm = [x+y for x,y in zip(w_norm, b_norm)]
#     else:
#         # w1_norm = [dump[k].abs().sum() 
#         #             for k in range(0, 4*time_steps) if k%4==0]
#         # b1_norm = [dump[k].abs().sum() 
#         #             for k in range(0, 4*time_steps) if k%4==1]
#         # w2_norm = [dump[k].abs().sum() 
#         #             for k in range(0, 4*time_steps) if k%4==2]
#         # b2_norm = [dump[k].abs().sum() 
#         #             for k in range(0, 4*time_steps) if k%4==3]
#         w1_norm = [np.linalg.norm(dump[k])
#                     for k in range(0, 4*time_steps) if k%4==0]
#         b1_norm = [np.linalg.norm(dump[k])
#                     for k in range(0, 4*time_steps) if k%4==1]
#         w2_norm = [np.linalg.norm(dump[k]) 
#                     for k in range(0, 4*time_steps) if k%4==2]
#         b2_norm = [np.linalg.norm(dump[k])
#                     for k in range(0, 4*time_steps) if k%4==3]
#         w_norm = [x+y for x,y in zip(w1_norm, w2_norm)]
#         b_norm = [x+y for x,y in zip(b1_norm, b2_norm)]
#         ctrl_norm = [x+y for x,y in zip(w_norm, b_norm)]
    
#     f1 = interp1d(integration_time,
#                   ctrl_norm, 
#                   kind='cubic', 
#                   fill_value="interpolate")
#     _time = torch.linspace(0., T, 150)
    
#     ax = plt.gca()
#     ax.set_axisbelow(True)                              
#     plt.rc('text', usetex=True)
#     plt.rc('font', family='serif')
#     plt.xlabel(r'$t_k$ ($k$ is a layer)')
#     plt.plot(_time, 
#              f1(_time), 
#              c='tab:blue', 
#              alpha=1, 
#              linewidth=2.5, 
#              label=r'$|u(t)|$')
    
#     if not hasattr(model, 'num_layers'):
#         ax.scatter(torch.linspace(0., T, len(ctrl_norm)), 
#                    ctrl_norm, 
#                    c='white', 
#                    alpha=1, 
#                    marker='.', 
#                    edgecolors='tab:blue')
#     #ax.legend(prop={'size':10}, loc="upper right", frameon=True)
#     ax.set_xlim([0, T])
#     ax.tick_params(axis='both', which='major', labelsize=8)
#     ax.tick_params(axis='both', which='minor', labelsize=6)

#     save_fig = 'controls.pdf'
#     if len(save_fig):
#         plt.savefig(save_fig, format='pdf', bbox_inches='tight')
#         plt.clf()
#         plt.close() 

# =============================================================================
# The resulting features
# =============================================================================

def feature_plot(feature_history, targets, alpha=0.9, filename='features.pdf'):
    base_filename = filename[:-4]
    
    ## We focus on 3 colors at most
    # For cross-entropy, as labels are in [m]
    if False in (t < 2 for t in targets): 
        color = ['mediumpurple' if targets[i] == 2.0 
                  else 'gold' if targets[i] == 0.0 
                  else 'mediumseagreen' 
                  for i in range(len(targets))]
        
    else:
        color = ['#3658BF' if targets[i] > 0.0 
                 else '#BF2633' 
                 for i in range(len(targets))]
    
    num_dims = feature_history[0].shape[1]
    features = feature_history[-1]
    i = len(feature_history)
    
    if num_dims == 2:
        ax = plt.gca()
        ax.set_facecolor('whitesmoke')                      
        #Gray background
        ax.set_axisbelow(True)                              
        #Axis beneath data
        ax.xaxis.grid(color='lightgray', linestyle='dotted')
        ax.yaxis.grid(color='lightgray', linestyle='dotted')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        label_size = 12
        plt.rcParams['xtick.labelsize'] = label_size
        plt.rcParams['ytick.labelsize'] = label_size 
        plt.xlabel(r'$x_1$', fontsize=13)
        plt.ylabel(r'$x_2$', fontsize=13)
        plt.scatter(features[:, 0].numpy(), 
                    features[:, 1].numpy(), 
                    c=color,
                    alpha=alpha, 
                    marker = 'o', 
                    linewidth=0.65, 
                    edgecolors='black')
        
    elif num_dims == 3:
        fig = plt.figure()
        ax = Axes3D(fig)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        label_size = 12
        plt.rcParams['xtick.labelsize'] = label_size
        plt.rcParams['ytick.labelsize'] = label_size
        ax.scatter(features[:, 0].numpy(), 
                   features[:, 1].numpy(), 
                   features[:, 2].numpy(),
                   c=color, 
                   alpha=alpha, 
                   marker = 'o', 
                   linewidth=0.65, 
                   edgecolors='black')
        plt.rc('grid', linestyle="dotted", color='lightgray')
        ax.grid(False)
        ax.view_init(elev=10)
        plt.locator_params(nbins=4)
        
    plt.savefig(base_filename + "{}.pdf".format(i), 
                format='pdf', 
                bbox_inches='tight')
    plt.clf()
    plt.close()

def plt_classifier(model, 
                   inputs, 
                   test_inputs, 
                   targets, 
                   test_targets, 
                   t=-1, 
                   plot_range=(-2.15, 2.15), 
                   num_steps=201, 
                   save_fig='generalization.pdf'):
    
    import matplotlib as mpl
    from matplotlib import rc
    from matplotlib.colors import ListedColormap

    rc("text", usetex = True)
    font = {'size'   : 13}
    rc('font', **font)
    
    ## We focus on 3 colors at most
    # For cross-entropy, as labels are in [m]
        
    if False in (t < 2 for t in targets): 
        plot_range = (-2.35, 2.35)
        color = ['mediumpurple' if targets[i] == 2.0 
                 else 'gold' if targets[i] == 0.0 
                 else 'mediumseagreen' 
                 for i in range(len(targets))]
        test_color = ['mediumpurple' if test_targets[i] == 2.0 
                      else 'gold' if test_targets[i] == 0.0 
                      else 'mediumseagreen' 
                      for i in range(len(test_targets))]
        cmap = mpl.cm.get_cmap("viridis_r")
        bounds = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    
    else:
        
        plot_range = (-1.01, 1.01)
        color = ['royalblue' if targets[i] < 0.0 
                 else 'crimson' 
                 for i in range(len(targets))]
        test_color = ['royalblue' if test_targets[i] < 0.0 
                      else 'crimson' 
                      for i in range(len(test_targets))]
        
        colors = [".0"]*5
        colors[0] = "#3a4bbf"
        colors[-1] = "#b30126"
        cmap = ListedColormap(colors=colors)

        if model.cross_entropy:
            bounds = [0.0, 0.1, 0.25, 0.35, 0.5, 0.65, 0.75, 0.9, 1.0]          
            # cross-entropy labels
        else: 
            print('Not cross entropy')
            bounds = [-1.0, -0.5, 0.0, 0.5, 1.0]
            # mse labels
    
    grid = torch.zeros((num_steps * num_steps, 2))
    idx = 0
    for x1 in np.linspace(plot_range[0], plot_range[1], num_steps):
        for x2 in np.linspace(plot_range[0], plot_range[1], num_steps):
            grid[idx, :] = torch.Tensor([x1, x2])
            idx += 1

    if not model.cross_entropy:
        #predictions, traj = model(grid)
        non_linearity = nn.Hardtanh(min_val=-1.0, max_val=1.0)
        _pre, traj = model(grid)
        a = 75
        predictions = non_linearity(a*traj[t])
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

    X_new, Y_new = np.meshgrid(_x,_y)
    i = plt.contourf(X_new, 
                     Y_new, 
                     pred_grid, 
                     vmin=vmin, 
                     vmax=vmax, 
                     cmap=cmap, 
                     norm=norm, 
                     alpha=1)

    plt.tick_params(axis='both', 
                    which='both', 
                    bottom=False, 
                    top=False,
                    labelbottom=False, 
                    right=False, 
                    left=False,
                    labelleft=False)
    
    plt.scatter(inputs[:,0], 
                inputs[:,1], 
                c=color, 
                alpha=0.9, 
                marker = 'o', 
                linewidth=0.45, 
                edgecolors='black', 
                label='train')
    
    plt.scatter(test_inputs[:,0], 
                test_inputs[:, 1],
                c=test_color, 
                alpha=0.9, 
                marker='o', 
                linewidth=1.75, 
                edgecolors='black', 
                label='test')
    
    for c in i.collections:
        c.set_edgecolor("face")
    
    from matplotlib.lines import Line2D
    if t==-1:
        #cb = fig.colorbar(i)
        #cb.ax.tick_params(size=0)
        legend_elements = [Line2D([0], 
                                  [0], 
                                  marker='o', 
                                  color='w', 
                                  mew=0.45, 
                                  mec='black', 
                                  label='train',
                                  markerfacecolor='lightgray',
                                  markersize=7),
                           Line2D([0], 
                                  [0], 
                                  marker='o', 
                                  color='w', 
                                  mew=1.75,
                                  mec='black', 
                                  label='test',
                                  markerfacecolor='lightgray', 
                                  markersize=7)]

        plt.legend(handles=legend_elements, 
                   loc="upper left", 
                   bbox_to_anchor=(-0.315, 1.025), 
                   frameon=False)
    
        plt.title('Generalization outside training data', fontsize=13)
    plt.xlabel(r'$x_1$', fontsize=13)
    plt.ylabel(r'$x_2$', fontsize=13)

    if len(save_fig):
        
        plt.savefig('generalization_%d.pdf' % t, 
                    format='pdf', 
                    bbox_inches='tight')
        plt.clf()
        plt.close()
        
def get_feature_history(trainer, 
                        dataloader, 
                        inputs, 
                        targets, 
                        num_epochs):
    
    feature_history = []
    features, _ = trainer.model(inputs, return_features=True)
    feature_history.append(features.detach())

    for i in range(num_epochs):
        trainer.train(dataloader, 1)
        features, _ = trainer.model(inputs, return_features=True)
        feature_history.append(features.detach())
    return feature_history

def histories_plt(all_history_info, 
                  plot_type='loss', 
                  shaded_err=False,
                  labels=[], 
                  include_mean=True, 
                  time_per_epoch=[], 
                  save_fig=''):

    rc("text", usetex = True)
    font = {'size'   : 13}
    rc('font', **font)

    for i, history_info in enumerate(all_history_info):
        
        color = 'tab:pink'
        color_val = 'tab:blue'
        if plot_type == 'loss':
            histories = history_info["epoch_loss_history"]
            histories_val = history_info["epoch_loss_val_history"]
        elif plot_type == 'acc':
            histories = history_info["epoch_acc_history"]
            histories_val = history_info["epoch_acc_val_history"]

        if len(time_per_epoch):
            xlabel = "Time (seconds)"
        else:
            xlabel = "Epochs"

        if include_mean:
            ax = plt.gca()
            ax.set_facecolor('whitesmoke')
            ax.set_axisbelow(True)    #Axis beneath data
            ax.xaxis.grid(color='lightgray', linestyle='dotted')
            ax.yaxis.grid(color='lightgray', linestyle='dotted')
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
                
            mean_history = np.array(histories).mean(axis=0)
            mean_history_val = np.array(histories_val).mean(axis=0)
            if len(time_per_epoch):
                epochs = time_per_epoch[i] * np.arange(len(histories[0]))
            else:
                epochs = list(range(len(histories[0])))

            if shaded_err:
                std_history = np.array(histories).std(axis=0)
                std_history_val = np.array(histories_val).std(axis=0)
                plt.fill_between(epochs, 
                                 mean_history - std_history,
                                 mean_history + std_history, 
                                 facecolor=color,
                                 alpha=0.5)
                plt.fill_between(epochs, 
                                 mean_history_val - std_history_val,
                                 mean_history_val + std_history_val, 
                                 facecolor=color_val,
                                 alpha=0.5)
                
            else:
                for k in range(len(histories)):
                    plt.plot(epochs, histories[k], c=color, alpha=0.1)
                    plt.plot(epochs, histories_val[k], c=color_val, alpha=0.1)

            plt.plot(epochs, mean_history, c=color, label="Train")
            plt.plot(epochs, mean_history_val, c=color_val, label="Test")
            ax.legend(prop={'size': 10}, loc="lower left", frameon=True)
            ax.set_xlim([0, len(epochs)-1])
            plt.xticks(range(0, len(epochs), len(epochs)//10), 
                       range(1, len(epochs)+1, len(epochs)//10))
        else:
            for k in range(len(histories)):
                plt.plot(histories[k], c=color, alpha=0.1)     
                plt.plot(histories_val[k], c=color_val, alpha=0.1) 
    
    plt.xlabel(xlabel)

    mnist = True
    if plot_type == "acc" and mnist:
        plt.ylim((0.75, 1.0))
        plt.title('Accuracy')
    else:
        plt.title('Error')

    if len(save_fig):
        plt.savefig(save_fig, format='pdf', bbox_inches='tight')
        plt.clf()
        plt.close()
