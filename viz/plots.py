import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.colors import LinearSegmentedColormap

categorical_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
all_categorical_colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                          '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                          '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                          '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

##--------------#
T = 15.0
time_steps = 30
dt = T/time_steps
integration_time = torch.linspace(0., T, time_steps)
##--------------#

def single_feature_plt(features, targets, save_fig=''):
    alpha = 0.9
    color = ['red' if targets[i, 0] > 0.0 else 'blue' for i in range(len(targets))]
    num_dims = features.shape[1]

    if num_dims == 2:
        plt.set_facecolor('whitesmoke')
        plt.title('Training points')
        plt.scatter(features[:, 0].numpy(), features[:, 1].numpy(), c=color,
                    alpha=alpha, marker = 'o', linewidths=0)
        ax = plt.gca()
    elif num_dims == 3:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.title('Training points')
        ax.scatter(features[:, 0].numpy(), features[:, 1].numpy(),
                   features[:, 2].numpy(), c=color, alpha=alpha,
                   linewidths=0, s=80)
    if len(save_fig):
        plt.savefig(save_fig, format='pdf', bbox_inches='tight')
        plt.clf()
        plt.close()
    else:
        plt.show()
##--------------#

##--------------#
def multi_feature_plt(features, targets, save_fig=''):
    alpha = 0.5
    color = ['tab:red' if targets[i, 0] > 0.0 else 'tab:blue' for i in range(len(targets))]
    num_dims = features[0].shape[1]

    if num_dims == 2:
        fig, axarr = plt.subplots(1, len(features), figsize=(20, 10))
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.xlabel(r'$x_{i, 1}$')
        plt.ylabel(r'$x_{i, 2}$')
        plt.title(r'Features $x_i(T)$ with $T=${}'.format(str(T)))   
        for i in range(len(features)):
            axarr[i].scatter(features[i][:, 0].numpy(), features[i][:, 1].numpy(),
                             c=color, alpha=alpha, linewidths=0)
            axarr[i].set_aspect(get_square_aspect_ratio(axarr[i]))
    elif num_dims == 3:
        fig = plt.figure(figsize=(20, 10))
        for i in range(len(features)):
            ax = fig.add_subplot(1, len(features), i + 1, projection='3d')

            ax.scatter(features[i][:, 0].numpy(), features[i][:, 1].numpy(),
                       features[i][:, 2].numpy(), c=color, alpha=alpha,
                       linewidths=0, s=80)
            ax.set_aspect(get_square_aspect_ratio(ax))
    fig.subplots_adjust(wspace=0.01)

    if len(save_fig):
        plt.savefig(save_fig, format='pdf', bbox_inches='tight')
        plt.clf()
        plt.close()
    else:
        plt.show()
##--------------#

##--------------#
def plt_state_component(model, inputs, targets, timesteps, component, highlight_inputs=False, save_fig='first.pdf'):

    from matplotlib import rc
    rc("text", usetex = True)
    font = {'size'   : 18}
    rc('font', **font)
    
    alpha = 0.75

    if hasattr(self.model, 'num_layers'):
        #ends, trajectories = model(inputs)
        #trajectories = trajectories.detach()
    
        ## ResNet (MNIST):
        ends, _, trajectories = model(inputs)
        trajectories = np.asarray(trajectories)
        #color = ..
    else:
    ## nODE (spheres):
        color = ['red' if targets[i, 0] > 0.0 else 'blue' for i in range(len(targets))]
        trajectories = model.odeblock.trajectory(inputs, timesteps).detach()
    

    inputs_aug = inputs
    
    for i in range(inputs_aug.shape[0]):
        ## ResNet:
        #y_traj = [x[i][component].detach().numpy() for x in trajectories]
        ## nODE:
        trajectory = trajectories[:, i, :]
        y_traj = trajectory[:, component].numpy()

        ax = plt.gca()
        ax.set_facecolor('whitesmoke')
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.title(r'Component of $\mathbf{x}_{i}(t)$', fontsize=12)
        #plt.title(r'$\tanh(P\mathbf{x}_{i}(t))$', fontsize=12)
        plt.xlabel(r'$t$ (layers)')
        #plt.plot(integration_time, y_traj, c='blue', alpha=alpha, linewidth=0.75)
        plt.plot(integration_time, y_traj, c=color[i], alpha=alpha, linewidth=0.75)
        ax.set_xlim([0, T])
        plt.rc('grid', linestyle="dotted", color='lightgray')
        ax.grid('on')
    
    if len(save_fig):
        plt.savefig(save_fig, format='pdf', bbox_inches='tight')
        plt.clf()
        plt.close()
##--------------#

##--------------#
##.. The norm of x(t)
def plt_norm_state(model, inputs, targets, timesteps, highlight_inputs=False, save_fig='norm_state.pdf'):

    from matplotlib import rc
    rc("text", usetex = True)
    font = {'size'   : 18}
    rc('font', **font)
    
    alpha = 0.9
    ## ResNet:
    #ends, _, traj = model(inputs)
    #traj = np.asarray(traj)
    #_ = np.asarray(_)
    #traj -> (40, 256, 784)
    #x_norm = [torch.norm(traj[k]) for k in range(timesteps)]
    #x_proj_norm = [torch.norm(_[k]) for k in range(timesteps)]
    
    ##nODE:
    ends, _ = model(inputs)
    _ = _.detach()
    
    ## ResNet:
    #loss = nn.CrossEntropyLoss()
    ## nODE:
    loss = nn.MSELoss()
    error = [loss(_[k], targets) for k in range(timesteps)]

    ax = plt.gca()
    ax.set_facecolor('whitesmoke')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.title(r'Decay of training error', fontsize=12)
    plt.xlabel(r'$t$ (layers)')
    #plt.ylabel(r'$|\mathbf{x}(t)|^2$')
    
    #plt.plot(integration_time, x_norm, c='crimson', alpha=alpha, linewidth=3, label=r'$|\mathbf{x}(t)|^2$')
    #plt.plot(integration_time, x_proj_norm, c='navy', alpha=alpha, linewidth=3, label=r'$|P\mathbf{x}(t)|^2$')
    
    # The training error
    plt.plot(integration_time, error, c='crimson', alpha=alpha, linewidth=2.25, label=r'$\phi(\mathbf{x}(t))$')
    ax.legend(prop={'size': 10}, fancybox=True, framealpha=0.2)
    ax.set_xlim([0, int(T)])
    plt.rc('grid', linestyle="dotted", color='lightgray')
    ax.grid('on')

    if len(save_fig):
        plt.savefig(save_fig, format='pdf', bbox_inches='tight')
        plt.clf()
        plt.close()  
##-------------# 

##--------------#
##.. The norm of x1 and x2(t)
def plt_norm_components(model, inputs, targets, timesteps, highlight_inputs=False, save_fig='norm.pdf'):

    from matplotlib import rc
    rc("text", usetex = True)
    font = {'size'   : 18}
    rc('font', **font)
    
    alpha = 0.9
    trajectories = model.odeblock.trajectory(inputs, timesteps).detach()
    ends, _ = model(inputs)
    _ = _.detach()
    
    input_dim = model.data_dim + model.augment_dim

    x_norm = [np.linalg.norm(trajectories[k, :, :], ord = 'fro') for k in range(timesteps)]
    _norm = [np.linalg.norm(_[k, :], ord = 'fro') for k in range(timesteps)]
    #x_norm_0 = [np.linalg.norm(trajectories[k, :, 0]) for k in range(timesteps)]
    #x_norm_1 = [np.linalg.norm(trajectories[k, :, 1]) for k in range(timesteps)]


    ax = plt.gca()
    ax.set_facecolor('whitesmoke')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.title('Stability of norms', fontsize=12)
    #plt.title(r'$t\mapsto|\mathbf{x}^j(t)|^2$')
    plt.xlabel(r'$t$ (layers)')
    #plt.ylabel(r'$|\mathbf{x}(t)|^2$')
    #plt.ylabel(r'$|x^j(t)|^2$')
    #plt.plot(integration_time, x_norm, c='blue', alpha=alpha, linewidth=2)
    plt.plot(integration_time, x_norm, c='cornflowerblue', alpha=alpha, linewidth=2.25, label=r'$|\mathbf{x}(t)|^2$')
    plt.plot(integration_time, _norm, c='darkorange', alpha=alpha, linewidth=2.25, label=r'$|P\mathbf{x}(t)|^2$')
    #plt.plot(integration_time, x_norm_0, c='teal', alpha=alpha, linewidth=3, label=r'$|\mathbf{x}^1(t)|^2$')
    #plt.plot(integration_time, x_norm_1, c='darkorange', alpha=alpha, linewidth=3, label=r'$|\mathbf{x}^2(t)|^2$')
    #if input_dim == 3:
    #    x_norm_2 = [np.linalg.norm(trajectories[k, :, 2]) for k in range(timesteps)]
    #    plt.plot(integration_time, x_norm_2, c='seagreen', alpha=alpha, linewidth=2, label=r'$|\mathbf{x}^3(t)|^2$')
    ax.legend(prop={'size': 10}, fancybox=True, framealpha=0.2)
    #plt.title(r'Norms of components of $\mathbf{x}(t)$', fontsize=12)
    #plt.title(r'Norms of $\mathbf{x}(t)$ and $P\mathbf{x}(t)$', fontsize=12)
    ax.set_xlim([0, T])
    plt.rc('grid', linestyle="dotted", color='lightgray')
    ax.grid('on')

    if len(save_fig):
        plt.savefig(save_fig, format='pdf', bbox_inches='tight')
        plt.clf()
        plt.close()  
##--------------# 

##--------------#
def input_space_plt(model, plot_range=(-2., 2.), num_steps=201, save_fig='generalization.pdf'):

    grid = torch.zeros((num_steps * num_steps, 2))
    idx = 0
    for x1 in np.linspace(plot_range[0], plot_range[1], num_steps):
        for x2 in np.linspace(plot_range[0], plot_range[1], num_steps):
            grid[idx, :] = torch.Tensor([x1, x2])
            idx += 1

    predictions, traj = model(grid)
    pred_grid = predictions.view(num_steps, num_steps).detach()

    colors = [(1, 1, 1), (0, 0, 1), (0.5, 0, 0.5), (1, 0, 0), (1, 1, 1)]
    colormap = LinearSegmentedColormap.from_list('cmap_red_blue', colors, N=256, gamma=1)

    #x = np.linspace(-2., 2., 201)
    #y = np.linspace(-2., 2., 201)
    #X, Y = np.meshgrid(x, y)

    # Plot input space as a heatmap
    #plt.imshow(pred_grid, vmin=-2., vmax=2., cmap=colormap, alpha=0.75)
    plt.imshow(pred_grid, vmin=-1.1, vmax=1.1, cmap='seismic', alpha=1)
    #plt.contourf(X, Y, pred_grid, 25, cmap=colormap)
    plt.colorbar()
    plt.tick_params(axis='both', which='both', bottom=False, top=False,
                        labelbottom=False, right=False, left=False,
                        labelleft=False)

    if len(save_fig):
        plt.savefig(save_fig, format='pdf', bbox_inches='tight')
        plt.clf()
        plt.close()
##--------------#

##--------------#
def histories_plt(all_history_info, plot_type='loss', shaded_err=False,
                  labels=[], include_mean=True, nfe_type='nfe',
                  time_per_epoch=[], save_fig=''):

    for i, history_info in enumerate(all_history_info):
        model_type = history_info["type"]
        if len(labels) > 0:
            color = categorical_colors[i % 4]
            label = labels[i]
        else:
            if model_type == 'resnet':
                color = categorical_colors[0]
                label = 'ResNet'
            if model_type == 'odenet':
                color = categorical_colors[1]
                label = 'Neural ODE'
            if model_type == 'anode':
                color = categorical_colors[2]
                #label = 'ANODE'
                label = 'Constant Controls'

        if model_type == 'resnet' and plot_type != 'loss':
            continue

        if plot_type == 'loss':
            histories = history_info["epoch_loss_history"]
            ylabel = "Loss"
        elif plot_type == 'nfe':
            if nfe_type == 'nfe':
                histories = history_info["epoch_nfe_history"]
            elif nfe_type == 'bnfe':
                histories = history_info["epoch_bnfe_history"]
            elif nfe_type == 'total_nfe':
                histories = history_info["epoch_total_nfe_history"]
            ylabel = "# of Function Evaluations"
        elif plot_type == 'nfe_vs_loss':
            histories_loss = history_info["epoch_loss_history"]
            if nfe_type == 'nfe':
                histories_nfe = history_info["epoch_nfe_history"]
            elif nfe_type == 'bnfe':
                histories_nfe = history_info["epoch_bnfe_history"]
            elif nfe_type == 'total_nfe':
                histories_nfe = history_info["epoch_total_nfe_history"]
            xlabel = "# of Function Evaluations"
            ylabel = "Loss"

        if plot_type == 'loss' or plot_type == 'nfe':
            if len(time_per_epoch):
                xlabel = "Time (seconds)"
            else:
                xlabel = "Epochs"

            if include_mean:
                mean_history = np.array(histories).mean(axis=0)
                if len(time_per_epoch):
                    epochs = time_per_epoch[i] * np.arange(len(histories[0]))
                else:
                    epochs = list(range(len(histories[0])))

                if shaded_err:
                    std_history = np.array(histories).std(axis=0)
                    plt.fill_between(epochs, mean_history - std_history,
                                     mean_history + std_history, facecolor=color,
                                     alpha=0.5)
                else:
                    for history in histories:
                        plt.plot(epochs, history, c=color, alpha=0.1)

                plt.plot(epochs, mean_history, c=color, label=label)
            else:
                for history in histories:
                    plt.plot(history, c=color, alpha=0.1)
        else:
            for j in range(len(histories_loss)):
                if j == 0:  # This is hacky, only used to avoid repeated labels
                    plt.scatter(histories_nfe[j], histories_loss[j], c=color,
                                alpha=0.5, label=label, linewidths=0)
                else:
                    plt.scatter(histories_nfe[j], histories_loss[j], c=color,
                                alpha=0.5, linewidths=0)

    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(bottom=0)

    if len(save_fig):
        plt.savefig(save_fig, format='pdf', bbox_inches='tight')
        plt.clf()
        plt.close()

##--------------#
def get_feature_history(trainer, dataloader, inputs, targets, num_epochs):
    
    feature_history = []
    features, _ = trainer.model(inputs, return_features=True)
    feature_history.append(features.detach())

    for i in range(num_epochs):
        trainer.train(dataloader, 1)
        features, _ = trainer.model(inputs, return_features=True)
        feature_history.append(features.detach())

    return feature_history

def get_square_aspect_ratio(plt_axis):
    return np.diff(plt_axis.get_xlim())[0] / np.diff(plt_axis.get_ylim())[0]
