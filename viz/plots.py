import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d


categorical_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

all_categorical_colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                          '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                          '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                          '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

T = 36
time_steps = int(pow(T, 1.5))
#dt = T/time_steps
integration_time = torch.linspace(0., T, time_steps)


def vector_field_plt(odefunc, num_points, timesteps, inputs=None, targets=None,
                     model=None, h_min=-2., h_max=2., t_max=1., extra_traj=[],
                     save_fig=''):
    t, hidden, dtdt, dhdt = ode_grid(odefunc, num_points, timesteps,
                                     h_min=h_min, h_max=h_max, t_max=t_max)
    t_grid, h_grid = np.meshgrid(t, hidden, indexing='ij')
    plt.quiver(t_grid, h_grid, dtdt, dhdt, width=0.004, alpha=0.6)

    if inputs is not None:
        if targets is not None:
            color = ['tab:red' if targets[i, 0] > 0 else 'tab:blue' for i in range(len(targets))]
        else:
            color = 'tab:red'
        plt.scatter(x=[0] * len(inputs), y=inputs[:, 0].numpy(), c=color, s=80)

    if targets is not None:
        color = ['tab:red' if targets[i, 0] > 0 else 'tab:blue' for i in range(len(targets))]
        plt.scatter(x=[t_max] * len(targets), y=targets[:, 0].numpy(), c=color,
                    s=80)

    if model is not None and inputs is not None:
        color = ['tab:red' if targets[i, 0] > 0 else 'tab:blue' for i in range(len(targets))]
        for i in range(len(inputs)):
            init_point = inputs[i:i+1]
            trajectory = model.trajectory(init_point, timesteps)
            plt.plot(t, trajectory[:, 0, 0].detach().numpy(), c=color[i],
                     linewidth=2)

    if len(extra_traj):
        for traj, color in extra_traj:
            num_steps = len(traj)
            t_traj = [t_max * float(i) / (num_steps - 1) for i in range(num_steps)]
            plt.plot(t_traj, traj, c=color, linestyle='--', linewidth=2)
            plt.scatter(x=t_traj[1:], y=traj[1:], c=color, s=20)

    plt.xlabel("t")
    plt.ylabel("h(t)")

    if len(save_fig):
        plt.savefig(save_fig, format='pdf', bbox_inches='tight')
        plt.clf()
        plt.close()


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


def single_feature_plt(features, targets, save_fig=''):
    alpha = 0.9
    color = ['tab:red' if targets[i, 0] > 0.0 else 'tab:blue' for i in range(len(targets))]
    num_dims = features.shape[1]

    if num_dims == 2:
        plt.title('Training points')
        plt.scatter(features[:, 0].numpy(), features[:, 1].numpy(), c=color,
                    alpha=alpha, marker = 'o', linewidths=0)
        #plt.tick_params(axis='both', which='both', bottom=False, top=False,
        #                labelbottom=False, right=False, left=False,
        #                labelleft=False)
        ax = plt.gca()
    elif num_dims == 3:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.title('Training points')
        ax.scatter(features[:, 0].numpy(), features[:, 1].numpy(),
                   features[:, 2].numpy(), c=color, alpha=alpha,
                   linewidths=0, s=80)
        #ax.tick_params(axis='both', which='both', bottom=False, top=False,
        #               labelbottom=False, right=False, left=False,
        #               labelleft=False)

    ax.set_aspect(get_square_aspect_ratio(ax))

    if len(save_fig):
        plt.savefig(save_fig, format='pdf', bbox_inches='tight')
        plt.clf()
        plt.close()
    else:
        plt.show()


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
            #axarr[i].tick_params(axis='both', which='both', bottom=False,
            #                     top=False, labelbottom=False, right=False,
            #                     left=False, labelleft=False)
            axarr[i].set_aspect(get_square_aspect_ratio(axarr[i]))
    elif num_dims == 3:
        fig = plt.figure(figsize=(20, 10))
        for i in range(len(features)):
            ax = fig.add_subplot(1, len(features), i + 1, projection='3d')

            ax.scatter(features[i][:, 0].numpy(), features[i][:, 1].numpy(),
                       features[i][:, 2].numpy(), c=color, alpha=alpha,
                       linewidths=0, s=80)
            #ax.tick_params(axis='both', which='both', bottom=False, top=False,
            #               labelbottom=False, right=False, left=False,
            #               labelleft=False)
            ax.set_aspect(get_square_aspect_ratio(ax))

    fig.subplots_adjust(wspace=0.01)

    if len(save_fig):
        plt.savefig(save_fig, format='pdf', bbox_inches='tight')
        plt.clf()
        plt.close()
    else:
        plt.show()


def trajectory_plt(model, inputs, targets, timesteps, highlight_inputs=False,
                   include_arrow=False, save_fig=''):
    alpha = 0.5
    color = ['tab:red' if targets[i, 0] > 0.0 else 'tab:blue' for i in range(len(targets))]
    trajectories = model.odeblock.trajectory(inputs, timesteps).detach()
    features = trajectories[-1]

    if model.augment_dim > 0:
        aug = torch.zeros(inputs.shape[0], model.odeblock.odefunc.augment_dim)
        inputs_aug = torch.cat([inputs, aug], 1)
    else:
        inputs_aug = inputs

    input_dim = model.data_dim + model.augment_dim

    if input_dim == 2:
        input_linewidths = 2 if highlight_inputs else 0
        plt.scatter(inputs_aug[:, 0].numpy(), inputs_aug[:, 1].numpy(), c=color,
                    alpha=alpha, linewidths=input_linewidths, edgecolor='orange')
        plt.scatter(features[:, 0].numpy(), features[:, 1].numpy(), c=color,
                    alpha=alpha, linewidths=0)

        for i in range(inputs_aug.shape[0]):
            trajectory = trajectories[:, i, :]
            x_traj = trajectory[:, 0].numpy()
            y_traj = trajectory[:, 1].numpy()
            plt.plot(x_traj, y_traj, c=color[i], alpha=alpha)
            if include_arrow:
                arrow_start = x_traj[-2], y_traj[-2]
                arrow_end = x_traj[-1], y_traj[-1]
                plt.arrow(arrow_start[0], arrow_start[1],
                          arrow_end[0] - arrow_start[0],
                          arrow_end[1] - arrow_start[1], shape='full', lw=0,
                          length_includes_head=True, head_width=0.15,
                          color=color[i], alpha=alpha)

        plt.tick_params(axis='both', which='both', bottom=False, top=False,
                        labelbottom=False, right=False, left=False,
                        labelleft=False)

        ax = plt.gca()
    elif input_dim == 3:
        # Create figure
        fig = plt.figure()
        ax = Axes3D(fig)

        input_linewidths = 1 if highlight_inputs else 0
        ax.scatter(inputs_aug[:, 0].numpy(), inputs_aug[:, 1].numpy(),
                   inputs_aug[:, 2].numpy(), c=color, alpha=alpha,
                   linewidths=input_linewidths, edgecolor='orange')
        ax.scatter(features[:, 0].numpy(), features[:, 1].numpy(),
                   features[:, 2].numpy(), c=color, alpha=alpha, linewidths=0)

        for i in range(inputs_aug.shape[0]):
            trajectory = trajectories[:, i, :]
            x_traj = trajectory[:, 0].numpy()
            y_traj = trajectory[:, 1].numpy()
            z_traj = trajectory[:, 2].numpy()
            ax.plot(x_traj, y_traj, z_traj, c=color[i], alpha=alpha)

            if include_arrow:
                arrow_start = x_traj[-2], y_traj[-2], z_traj[-2]
                arrow_end = x_traj[-1], y_traj[-1], z_traj[-1]

                arrow = Arrow3D([arrow_start[0], arrow_end[0]],
                                [arrow_start[1], arrow_end[1]],
                                [arrow_start[2], arrow_end[2]],
                                mutation_scale=15,
                                lw=0, color=color[i], alpha=alpha)
                ax.add_artist(arrow)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    else:
        raise RuntimeError("Input dimension must be 2 or 3 but was {}".format(input_dim))

    ax.set_aspect(get_square_aspect_ratio(ax))

    if len(save_fig):
        plt.savefig(save_fig, format='pdf', bbox_inches='tight')
        plt.clf()
        plt.close()


def plt_x_component(model, inputs, targets, timesteps, highlight_inputs=False, save_fig='first.pdf'):
    
    from matplotlib import rc
    rc("text", usetex = True)
    font = {'size'   : 18}
    rc('font', **font)
    
    alpha = 0.5
    color = ['tab:red' if targets[i, 0] > 0.0 else 'tab:blue' for i in range(len(targets))]
    trajectories = model.odeblock.trajectory(inputs, timesteps).detach()
    features = trajectories[-1]

    if model.augment_dim > 0:
        aug = torch.zeros(inputs.shape[0], model.odeblock.odefunc.augment_dim)
        inputs_aug = torch.cat([inputs, aug], 1)
    else:
        inputs_aug = inputs

    input_dim = model.data_dim + model.augment_dim

    if input_dim == 2:
        input_linewidths = 2 if highlight_inputs else 0
        
        for i in range(inputs_aug.shape[0]):
            trajectory = trajectories[:, i, :]
            x_traj = trajectory[:, 0].numpy()
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            plt.title(r'$\mathbf{x}_{i, 1}(t)$')
            plt.xlabel(r'$t$ (layers)')
            plt.plot(integration_time, x_traj, c=color[i], alpha=alpha, linewidth=0.75)
            
            ax = plt.gca()
            ax.set_xlim([0, T])

        ax = plt.gca()
    ax.set_aspect(get_square_aspect_ratio(ax))
    
    if len(save_fig):
        plt.savefig(save_fig, format='pdf', bbox_inches='tight')
        plt.clf()
        plt.close()

def plt_y_component(model, inputs, targets, timesteps, highlight_inputs=False, save_fig='second.pdf'):

    from matplotlib import rc
    rc("text", usetex = True)
    font = {'size'   : 18}
    rc('font', **font)
    
    alpha = 0.5
    color = ['tab:red' if targets[i, 0] > 0.0 else 'tab:blue' for i in range(len(targets))]
    trajectories = model.odeblock.trajectory(inputs, timesteps).detach()
    features = trajectories[-1]

    if model.augment_dim > 0:
        aug = torch.zeros(inputs.shape[0], model.odeblock.odefunc.augment_dim)
        inputs_aug = torch.cat([inputs, aug], 1)
    else:
        inputs_aug = inputs

    input_dim = model.data_dim + model.augment_dim

    if input_dim == 2:
        input_linewidths = 2 if highlight_inputs else 0
        
        for i in range(inputs_aug.shape[0]):
            trajectory = trajectories[:, i, :]
            y_traj = trajectory[:, 1].numpy()
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            plt.title(r'$\mathbf{x}_{i, 2}(t)$')
            plt.xlabel(r'$t$ (layers)')
            plt.plot(integration_time, y_traj, c=color[i], alpha=alpha, linewidth=0.75)
            
            ax = plt.gca()
            ax.set_xlim([0, T])
        
        ax = plt.gca()
    ax.set_aspect(get_square_aspect_ratio(ax))
    
    if len(save_fig):
        plt.savefig(save_fig, format='pdf', bbox_inches='tight')
        plt.clf()
        plt.close()


def input_space_plt(model, plot_range=(-2., 2.), num_steps=201, save_fig='gen.pdf'):

    grid = torch.zeros((num_steps * num_steps, 2))
    idx = 0
    for x1 in np.linspace(plot_range[0], plot_range[1], num_steps):
        for x2 in np.linspace(plot_range[0], plot_range[1], num_steps):
            grid[idx, :] = torch.Tensor([x1, x2])
            idx += 1

    predictions, traj = model(grid)
    pred_grid = predictions.view(num_steps, num_steps).detach()

    colors = [(1, 1, 1), (0, 0, 1), (0.5, 0, 0.5), (1, 0, 0), (1, 1, 1)]
    colormap = LinearSegmentedColormap.from_list('cmap_red_blue', colors, N=300)

    # Plot input space as a heatmap
    plt.imshow(pred_grid, vmin=-2., vmax=2., cmap=colormap, alpha=0.75)
    plt.colorbar()
    plt.tick_params(axis='both', which='both', bottom=False, top=False,
                        labelbottom=False, right=False, left=False,
                        labelleft=False)

    if len(save_fig):
        #plt.savefig(save_fig, format='png', dpi=400, bbox_inches='tight')
        plt.savefig(save_fig, format='pdf', bbox_inches='tight')
        plt.clf()
        plt.close()


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def ode_grid(odefunc, num_points, timesteps, h_min=-2., h_max=2., t_max=1.):
    
    t = np.linspace(0., t_max, timesteps)
    hidden = np.linspace(h_min, h_max, num_points)
    dtdt = np.ones((timesteps, num_points)) 
    dhdt = np.zeros((timesteps, num_points))
    for i in range(len(t)):
        for j in range(len(hidden)):
            h_j = torch.Tensor([hidden[j]]).unsqueeze(0)
            dhdt[i, j] = odefunc(t[i], h_j)
    return t, hidden, dtdt, dhdt


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
