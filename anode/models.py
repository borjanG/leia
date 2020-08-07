import torch
import torch.nn as nn
from math import pi
from torchdiffeq import odeint, odeint_adjoint
from math import pi, sin, sqrt, cos
from functools import reduce
from operator import mul

MAX_NUM_STEPS = 1000    #.. Maximum number of steps for ODE solver
T = 16.0                #.. The time horizon
num_steps = int(pow(T, 1.5))
dt = T/pow(T, 1.5)

class ODEFunc(nn.Module):
    """
    The nonlinear right hand side $f(u(t), x(t)) of the ODE.
    """

    def __init__(self, device, data_dim, hidden_dim, augment_dim=0, non_linearity='tanh'):
        
        super(ODEFunc, self).__init__()

        self.device = device
        ##.. Dimensions
        self.augment_dim = augment_dim
        self.data_dim = data_dim
        self.input_dim = data_dim + augment_dim
        self.hidden_dim = hidden_dim
        
        self.nfe = 0 

        ##.. Activation functions
        if non_linearity == 'relu':
            self.non_linearity = nn.ReLU(inplace=True)
        else:
            self.non_linearity = nn.Tanh()

        ##.. Time independent controls
        
        #self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        #self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        #self.fc3 = nn.Linear(hidden_dim, self.input_dim)

        ##.. Time dependent controls

        self.filtering = time_steps

        ##.. Recall that x = nn.Linear(n, m) provides an array x.weight
        ##.. which has m rows and n columns.

        ##.. R^{d_aug} -> R^{d_hid} layer 
        #self.fc1_time = nn.Linear(self.input_dim, hidden_dim*self.filtering)
        ##.. R^{d_hid} -> R^{d_hid} layer
        self.fc2_time = nn.Linear(hidden_dim, hidden_dim*self.filtering)
        ##.. R^{d_hid} -> R^{d_aug} layer
        #self.fc3_time = nn.Linear(hidden_dim, self.input_dim*self.filtering)

    def forward(self, t, x):

        self.nfe += 1
        weights = self.fc2_time.weight
        biases = self.fc2_time.bias

        out = self.non_linearity(x)

        dt = T/(self.filtering)
        
        k = int(t/dt)
        At = weights[k*self.input_dim:(k+1)*self.input_dim] 
        bt = biases[k*self.input_dim:(k+1)*self.input_dim]

        # times = torch.linspace(0., T, self.filtering)
        # lagrange = lambda t, j: reduce(mul, [(t-times[m])/(times[j]-times[m]) for m in range(self.filtering) if m!=j],1)

        # At = weights[:self.input_dim]*lagrange(t, 0)
        # bt = biases[:self.input_dim]*lagrange(t, 0)
        # for j in range(1, self.filtering):
        #     At += weights[j*self.input_dim:(j+1)*self.input_dim]*lagrange(t,j)
        #     bt += biases[j*self.input_dim:(j+1)*self.input_dim]*lagrange(t,j)

        out = out.matmul(At.t())+bt

        return out

class ODEBlock(nn.Module):

    def __init__(self, device, odefunc, is_conv=False, tol=1e-3, adjoint=False):
        super(ODEBlock, self).__init__()
        self.adjoint = adjoint
        self.device = device
        self.is_conv = is_conv
        self.odefunc = odefunc
        self.tol = tol

    def forward(self, x, eval_times=None):
        self.odefunc.nfe = 0

        if eval_times is None:
            integration_time = torch.tensor([0, T]).float().type_as(x)
        else:
            integration_time = eval_times.type_as(x)

        if self.odefunc.augment_dim > 0:
            if self.is_conv:
                batch_size, channels, height, width = x.shape
                aug = torch.zeros(batch_size, self.odefunc.augment_dim,
                                  height, width).to(self.device)
                x_aug = torch.cat([x, aug], 1)
            else:
                aug = torch.zeros(x.shape[0], self.odefunc.augment_dim).to(self.device)
                x_aug = torch.cat([x, aug], 1)
        else:
            x_aug = x

        if self.adjoint:
            # out = odeint_adjoint(self.odefunc, x_aug, integration_time,
            #                      rtol=self.tol, atol=self.tol, method='dopri5',
            #                      options={'max_num_steps': MAX_NUM_STEPS})

            out = odeint_adjoint(self.odefunc, x_aug, integration_time, method='euler', options={'step_size': T/time_steps})
        else:
            # out = odeint(self.odefunc, x_aug, integration_time,
            #              rtol=self.tol, atol=self.tol, method='dopri5',
            #              options={'max_num_steps': MAX_NUM_STEPS})
            out = odeint(self.odefunc, x_aug, integration_time, method='euler', options={'step_size': T/time_steps})

        if eval_times is None:
            return out[1] 
        else:
            return out

    def trajectory(self, x, timesteps):
        integration_time = torch.linspace(0., T, timesteps)
        return self.forward(x, eval_times=integration_time)


class ODENet(nn.Module):
    def __init__(self, device, data_dim, hidden_dim, output_dim=1,
                 augment_dim=0, non_linearity='tanh',
                 tol=1e-3, adjoint=False):
        super(ODENet, self).__init__()
        self.device = device
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.augment_dim = augment_dim
        self.output_dim = output_dim
        self.tol = tol

        odefunc = ODEFunc(device, data_dim, hidden_dim, augment_dim, non_linearity)

        self.odeblock = ODEBlock(device, odefunc, tol=tol, adjoint=adjoint)
        self.linear_layer = nn.Linear(self.odeblock.odefunc.input_dim,
                                         self.output_dim)

        #### When the system evolves in hidden_dim, use this and comment the above self.linear_layer
        #### self.linear_layer = nn.Linear(self.hidden_dim, self.output_dim)
        self.non_linearity = nn.Tanh()

    def forward(self, x, return_features=False):

        features = self.odeblock(x)
        pred = self.linear_layer(features)
        pred = self.non_linearity(pred)

        self.traj = self.odeblock.trajectory(x, time_steps)
        self.proj_traj = self.linear_layer(self.traj)

        if return_features:
            return features, pred
        return pred, self.proj_traj
        #return pred, self.traj
