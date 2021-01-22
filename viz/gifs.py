import imageio
import torch
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

##--------------#
T = 15.0
time_steps = 15
dt = T/time_steps
##--------------#

def feature_evolution_gif(feature_history, targets, dpi=30, alpha=0.9,
                          filename='feature_evolution.gif'):
    
    if not filename.endswith(".gif"):
        raise RuntimeError("Filename must end in with .gif, but filename is {}".format(filename))
    base_filename = filename[:-4]

    #color = ['crimson' if targets[i, 0] > 0.0 else 'dodgerblue' for i in range(len(targets))]
    color = ['crimson' if targets[i] > 0.0 else 'dodgerblue' for i in range(len(targets))]
    
    #color = ['mediumpurple' if targets[i] == 2.0 else 'gold' if targets[i] == 0.0 else 'mediumseagreen' for i in range(len(targets))]
    
    
    #color = ['crimson' if targets[i, 0] == 0.0 else 'dodgerblue' if targets[i,0] == 1.0 else 'mediumseagreen' for i in range(len(targets))]
    #color = ['mediumpurple' if targets[i, 0] == 1.0 else 'gold' if targets[i,0] == -1.0 else 'mediumseagreen' for i in range(len(targets))]
    num_dims = feature_history[0].shape[1]

    features = feature_history[-1]
    i = len(feature_history)
    
    #for i, features in enumerate(feature_history):
    if num_dims == 2:
        ax = plt.gca()
        ax.set_facecolor('whitesmoke')
        ax.set_axisbelow(True)
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

        #plt.savefig(base_filename + "{}.png".format(i),
        #            format='png', dpi=dpi, bbox_inches='tight')
        
        #if i in [0, len(feature_history)//5, len(feature_history)//2, len(feature_history)-1]:
    plt.savefig(base_filename + "{}.pdf".format(i), format='pdf', bbox_inches='tight')
    plt.clf()
    plt.close()

#    imgs = []
#    #for i in range(len(feature_history)):
#        img_file = base_filename + "{}.png".format(i)
#        imgs.append(imageio.imread(img_file))
#        os.remove(img_file)  
#    imageio.mimwrite(filename, imgs)

def trajectory_gif(model, inputs, targets, timesteps, dpi=200, alpha=0.9,
                   alpha_line=1, filename='trajectory.gif'):
    
    from matplotlib import rc
    from scipy.interpolate import interp1d
    rc("text", usetex = True)
    font = {'size'   : 18}
    rc('font', **font)

    if not filename.endswith(".gif"):
        raise RuntimeError("Filename must end in with .gif, but filename is {}".format(filename))
    base_filename = filename[:-4]

    #color = ['crimson' if targets[i, 0] > 0.0 else 'dodgerblue' for i in range(len(targets))]
    #color = ['crimson' if targets[i] > 0.0 else 'dodgerblue' for i in range(len(targets))]    
    #color = ['mediumpurple' if targets[i] == 2.0 else 'gold' if targets[i] == 0.0 else 'mediumseagreen' for i in range(len(targets))]
    color = ['crimson' if targets[i] > 0.0 else 'dodgerblue' for i in range(len(targets))]
    
    #color = ['crimson' if targets[i, 0] == 0.0 else 'dodgerblue' if targets[i,0] == 1.0 else 'mediumseagreen' for i in range(len(targets))]
    #color = ['mediumpurple' if targets[i, 0] == 1.0 else 'gold' if targets[i,0] == -1.0 else 'mediumseagreen' for i in range(len(targets))]

    trajectories = model.odeblock.trajectory(inputs, timesteps).detach()
    num_dims = trajectories.shape[2]

    x_min, x_max = trajectories[:, :, 0].min(), trajectories[:, :, 0].max()
    y_min, y_max = trajectories[:, :, 1].min(), trajectories[:, :, 1].max()
    if num_dims == 3:
        z_min, z_max = trajectories[:, :, 2].min(), trajectories[:, :, 2].max()
    margin = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= margin * x_range
    x_max += margin * x_range
    y_min -= margin * y_range
    y_max += margin * y_range
    if num_dims == 3:
        z_range = z_max - z_min
        z_min -= margin * z_range
        z_max += margin * z_range
        
    integration_time = torch.linspace(0.0, T, timesteps)
    
    interp_x = []
    interp_y = []
    interp_z = []
    for i in range(inputs.shape[0]):
        interp_x.append(interp1d(integration_time, trajectories[:, i, 0], kind='cubic', fill_value='extrapolate'))
        interp_y.append(interp1d(integration_time, trajectories[:, i, 1], kind='cubic', fill_value='extrapolate'))
        if num_dims == 3:
            interp_z.append(interp1d(integration_time, trajectories[:, i, 2], kind='cubic', fill_value='extrapolate'))
    
    interp_time = 180
    _time = torch.linspace(0., T, interp_time)

    plt.rc('grid', linestyle="dotted", color='lightgray')
    #for t in range(timesteps):
    for t in range(interp_time):
        if num_dims == 2:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            label_size = 13
            plt.rcParams['xtick.labelsize'] = label_size
            plt.rcParams['ytick.labelsize'] = label_size 
            ax.set_axisbelow(True)
            ax.xaxis.grid(color='lightgray', linestyle='dotted')
            ax.yaxis.grid(color='lightgray', linestyle='dotted')
            ax.set_facecolor('whitesmoke')
            
            
            #plt.scatter(trajectories[t, :, 0].numpy(), trajectories[t, :, 1].numpy(), c=color,
            #            alpha=alpha, marker = 'o', linewidths=0)
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            plt.xlabel(r'$x_1$', fontsize=12)
            plt.ylabel(r'$x_2$', fontsize=12)
            plt.scatter([x(_time)[t] for x in interp_x], 
                         [y(_time)[t] for y in interp_y], 
                         c=color, alpha=alpha, marker = 'o', linewidth=0.65, edgecolors='black', zorder=3)

            if t > 0:
                for i in range(inputs.shape[0]):
                    x_traj = interp_x[i](_time)[:t+1]
                    y_traj = interp_y[i](_time)[:t+1]
                    
                    #trajectory = trajectories[:t + 1, i, :]
                    #x_traj = trajectory[:, 0].numpy()
                    #y_traj = trajectory[:, 1].numpy()
    
                    plt.plot(x_traj, y_traj, c=color[i], alpha=alpha_line, linewidth = 0.75, zorder=1)
            
        elif num_dims == 3:
            fig = plt.figure()
            ax = Axes3D(fig)
            label_size = 12
            plt.rcParams['xtick.labelsize'] = label_size
            plt.rcParams['ytick.labelsize'] = label_size 

            #ax.scatter(trajectories[t, :, 0].numpy(),
            #           trajectories[t, :, 1].numpy(),
            #           trajectories[t, :, 2].numpy(),
            #           c=color, alpha=alpha, marker = 'o', linewidth=0.65, edgecolors='black')
            
            ax.scatter([x(_time)[t] for x in interp_x], 
                        [y(_time)[t] for y in interp_y],
                        [z(_time)[t] for z in interp_z],
                        c=color, alpha=alpha, marker = 'o', linewidth=0.65, edgecolors='black')
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')           
            if t > 0:
                for i in range(inputs.shape[0]):
#                    trajectory = trajectories[:t + 1, i, :]
#                    x_traj = trajectory[:, 0].numpy()
#                    y_traj = trajectory[:, 1].numpy()
#                    z_traj = trajectory[:, 2].numpy()
                    
                    x_traj = interp_x[i](_time)[:t+1]
                    y_traj = interp_y[i](_time)[:t+1]
                    z_traj = interp_z[i](_time)[:t+1]
                    
                    ax.plot(x_traj, y_traj, z_traj, c=color[i], alpha=alpha_line, linewidth = 0.75)
                
            ax.set_xlim3d(x_min, x_max)
            ax.set_ylim3d(y_min, y_max)
            ax.set_zlim3d(z_min, z_max)
            
            plt.rc('grid', linestyle="dotted", color='lightgray')
            ax.grid(True)
            plt.locator_params(nbins=4)

        plt.savefig(base_filename + "{}.png".format(t),
                    format='png', dpi=dpi, bbox_inches='tight')
        # Save only 3 frames (.pdf for paper)
        if t in [0, interp_time//5, interp_time//2, interp_time-1]:
        #if t in [0, timesteps//5, timesteps//2, timesteps-1]:
            plt.savefig(base_filename + "{}.pdf".format(t), format='pdf', bbox_inches='tight')
        plt.clf()
        plt.close()

    imgs = []
    for i in range(interp_time):
        img_file = base_filename + "{}.png".format(i)
        imgs.append(imageio.imread(img_file))
        os.remove(img_file) 
    imageio.mimwrite(filename, imgs)

def mnist_gif(model, inputs, targets, timesteps, component, filename='mnist.gif'):

    from matplotlib import rc
    rc("text", usetex = True)
    font = {'size'   : 18}
    rc('font', **font)
    
    if not filename.endswith(".gif"):
        raise RuntimeError("Filename must end in with .gif, but filename is {}".format(filename))
    base_filename = filename[:-4]
    
    ends, _, traj = model(inputs)
    _ = np.asarray(_)

    
    ax = plt.gca()
    ax.set_facecolor('whitesmoke')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    for k in range(timesteps):
        plt.title(r't={}'.format(k))
        #plt.imsave('mnist{}.png'.format(k), traj[k].detach().numpy()[component].reshape(64,64), cmap='gray')
        #plt.imsave('mnist{}.pdf'.format(k), traj[k].detach().numpy()[component].reshape(64,64), cmap='gray', format='pdf')
        plt.imsave('mnist{}.png'.format(k), traj[k].detach().numpy()[component].reshape(28,28), cmap='gray')
        plt.imsave('mnist{}.pdf'.format(k), traj[k].detach().numpy()[component].reshape(28,28), cmap='gray', format='pdf')
    
    imgs = []
    for i in range(timesteps):
        img_file = base_filename + "{}.png".format(i)
        imgs.append(imageio.imread(img_file))
        os.remove(img_file) 
    imageio.mimwrite(filename, imgs) 