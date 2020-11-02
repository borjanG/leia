import imageio
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from viz.plots import get_square_aspect_ratio
import matplotlib.pyplot as plt

##--------------#
##.. Turnpike
T = 20.0
time_steps = 20
dt = T/time_steps

##.. Not Turnpike
#T = 81.0                
#time_steps = int(pow(T, 1.5))
#dt = T/pow(T, 1.5)
##--------------#

##--------------#
def feature_evolution_gif(feature_history, targets, dpi=150, alpha=0.9,
                          filename='feature_evolution.gif'):
    
    if not filename.endswith(".gif"):
        raise RuntimeError("Filename must end in with .gif, but filename is {}".format(filename))
    base_filename = filename[:-4]

    color = ['red' if targets[i, 0] > 0.0 else 'blue' for i in range(len(targets))]
    num_dims = feature_history[0].shape[1]

    for i, features in enumerate(feature_history):
        if num_dims == 2:
            ax = plt.gca()
            ax.set_facecolor('whitesmoke')
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            plt.xlabel(r'$\mathbf{x}_{i, 1}(t)$')
            plt.ylabel(r'$\mathbf{x}_{i, 2}(t)$')
            #plt.title(r'$T=${}'.format(str(T)))
            plt.scatter(features[:, 0].numpy(), features[:, 1].numpy(), c=color,
                        alpha=alpha, marker = 'o', linewidths=0)
        
        elif num_dims == 3:
            fig = plt.figure()
            ax = Axes3D(fig)
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            plt.xlabel(r'$\mathbf{x}_{i, 1}(t)$', fontsize=10)
            plt.ylabel(r'$\mathbf{x}_{i, 2}(t)$', fontsize=10)
            #plt.ylabel(r'$\mathbf{x}_{i, 3}(t)$')
            #plt.title(r'$T=${}'.format(str(T)))
            
            #ax.set_xticks([])
            #ax.set_yticks([])
            #ax.set_zticks([])

            ax.scatter(features[:, 0].numpy(), features[:, 1].numpy(), features[:, 1].numpy(),
                       c=color, alpha=alpha, marker = 'o', linewidths=0)
            
            ax.grid(b=False)
            plt.locator_params(nbins=4)

        plt.savefig(base_filename + "{}.png".format(i),
                    format='png', dpi=dpi, bbox_inches='tight')
        
        if i in [0, len(feature_history)//5, len(feature_history)//2, len(feature_history)-1]:
            plt.savefig(base_filename + "{}.pdf".format(i), format='pdf', bbox_inches='tight')
        plt.clf()
        plt.close()

    imgs = []
    for i in range(len(feature_history)):
        img_file = base_filename + "{}.png".format(i)
        imgs.append(imageio.imread(img_file))
        os.remove(img_file)  
    imageio.mimwrite(filename, imgs)
##--------------#

##--------------#
def trajectory_gif(model, inputs, targets, timesteps, dpi=150, alpha=1,
                   alpha_line=1, filename='trajectory.gif'):
    
    from matplotlib import rc
    rc("text", usetex = True)
    font = {'size'   : 18}
    rc('font', **font)

    if not filename.endswith(".gif"):
        raise RuntimeError("Filename must end in with .gif, but filename is {}".format(filename))
    base_filename = filename[:-4]

    color = ['red' if targets[i, 0] > 0.0 else 'blue' for i in range(len(targets))]

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

    for t in range(timesteps):
        if num_dims == 2:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_facecolor('whitesmoke')
            plt.scatter(trajectories[t, :, 0].numpy(), trajectories[t, :, 1].numpy(), c=color,
                        alpha=alpha, marker = 'o', linewidths=0)

            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            plt.xlabel(r'$\mathbf{x}_{i, 1}(t)$')
            plt.ylabel(r'$\mathbf{x}_{i, 2}(t)$')
            #plt.title(r'$T=${}'.format(str(T)))

            if t > 0:
                for i in range(inputs.shape[0]):
                    trajectory = trajectories[:t + 1, i, :]
                    x_traj = trajectory[:, 0].numpy()
                    y_traj = trajectory[:, 1].numpy()
                    plt.plot(x_traj, y_traj, c=color[i], alpha=alpha_line, linewidth = 0.75)
            
        elif num_dims == 3:
            fig = plt.figure()
            ax = Axes3D(fig)

            ax.scatter(trajectories[t, :, 0].numpy(),
                       trajectories[t, :, 1].numpy(),
                       trajectories[t, :, 2].numpy(),
                       c=color, alpha=alpha, marker = 'o', linewidths=0)
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            plt.xlabel(r'$\mathbf{x}_{i, 1}(t)$', fontsize=10)
            plt.ylabel(r'$\mathbf{x}_{i, 2}(t)$', fontsize=10)
            #plt.zlabel(r'$\mathbf{x}_{i, 3}(t)$')
            #plt.title(r'$T=${}'.format(str(T)))            
            if t > 0:
                for i in range(inputs.shape[0]):
                    trajectory = trajectories[:t + 1, i, :]
                    x_traj = trajectory[:, 0].numpy()
                    y_traj = trajectory[:, 1].numpy()
                    z_traj = trajectory[:, 2].numpy()
                    ax.plot(x_traj, y_traj, z_traj, c=color[i], alpha=alpha_line, linewidth = 0.75)
                    
            #ax.set_xticks([])
            #ax.set_yticks([])
            #ax.set_zticks([])

            ax.set_xlim3d(x_min, x_max)
            ax.set_ylim3d(y_min, y_max)
            ax.set_zlim3d(z_min, z_max)
            
            ax.grid(b=False)
            plt.locator_params(nbins=4)

        plt.savefig(base_filename + "{}.png".format(t),
                    format='png', dpi=dpi, bbox_inches='tight')
        # Save only 3 frames (.pdf for paper)
        if t in [0, timesteps//5, timesteps//2, timesteps-1]:
            plt.savefig(base_filename + "{}.pdf".format(t), format='pdf', bbox_inches='tight')
        plt.clf()
        plt.close()

    imgs = []
    for i in range(timesteps):
        img_file = base_filename + "{}.png".format(i)
        imgs.append(imageio.imread(img_file))
        os.remove(img_file) 
    imageio.mimwrite(filename, imgs)
##--------------#