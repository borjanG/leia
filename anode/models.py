import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint
MAX_NUM_STEPS = 1000

##--------------#
T = 15.0
time_steps = 30
dt = T/time_steps
##--------------#

class ODEFunc(nn.Module):
    """
    The nonlinear right hand side $f(u(t), x(t)) of the ODE.
    """
    def __init__(self, device, data_dim, hidden_dim, augment_dim=0, non_linearity='tanh'):
        super(ODEFunc, self).__init__()
        self.device = device
        self.augment_dim = augment_dim
        self.data_dim = data_dim
        self.input_dim = data_dim + augment_dim
        self.hidden_dim = hidden_dim
        self.nfe = 0 

        ##--------------#
        if non_linearity == 'relu':
            self.non_linearity = nn.ReLU(inplace=True)
        if non_linearity == 'leakyrelu':
            self.non_linearity = nn.LeakyReLU(negative_slope=0.25, inplace=True)
        if non_linearity == 'sigomoid':
            self.non_linearity = nn.Sigmoid()
        else:
            self.non_linearity = nn.Tanh()
        ##--------------#

        ##--------------#        
        ##.. Tunable projectors:
        #self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        #self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        #self.fc3 = nn.Linear(hidden_dim, 2)
        ##--------------#

        ##--------------#
        ##.. Time-dependent controls
        ##.. Recall that x = nn.Linear(n, m) provides an array x.weight
        ##.. which has m rows and n columns.
        ##.. R^{d_aug} -> R^{d_hid} layer 
        #self.fc1_time = nn.Linear(self.input_dim, time_steps)
        #self.fc1_time = nn.Linear(self.input_dim, hidden_dim*time_steps)
        ##.. R^{d_hid} -> R^{d_hid} layer
        self.fc2_time = nn.Linear(hidden_dim, hidden_dim*time_steps)
        ##.. R^{d_hid} -> R^{d_aug} layer
        #self.fc3_time = nn.Linear(hidden_dim, self.input_dim*time_steps)
        ##--------------#

    def forward(self, t, x):

        self.nfe += 1
        weights = self.fc2_time.weight
        biases = self.fc2_time.bias

        #---------------#
        ## In the case of Lin et al. '18 model
        #weights_1 = self.fc1_time.weight
        #weights_2 = self.fc3_time.weight
        #biases = self.fc1_time.bias
        #---------------#

        if t==0:
            return x
            ## In case of MNIST:
            #return x.view(x.size(0), -1)
        else:
            ##--------------#
            ## w(t)\sigma(x(t))+b(t)
            out = self.non_linearity(x)        
            k = int(t/dt)
            w_t = weights[k*self.hidden_dim : (k+1)*self.hidden_dim] 
            b_t = biases[k*self.hidden_dim : (k+1)*self.hidden_dim]
            out = out.matmul(w_t.t())+b_t
            
            ## \sigma(w(t)x(t)+b(t))
#            k = int(t/dt)
#            w_t = weights[k*self.hidden_dim : (k+1)*self.hidden_dim] 
#            b_t = biases[k*self.hidden_dim : (k+1)*self.hidden_dim]
#            out = x.matmul(w_t.t())+b_t
#            out = self.non_linearity(out)
            ##--------------#
            
            ##--------------#
            ## w1(t)\sigma(w2(t)x(t)+b2(t))+b1(t)
#            k = int(t/dt)
#            weights1 = self.fc1_time.weight
#            biases1 = self.fc1_time.bias
#            weights2 = self.fc3_time.weight
#            biases2 = self.fc3_time.bias
#            w1_t = weights1[k*self.hidden_dim : (k+1)*self.hidden_dim] 
#            b1_t = biases1[k*self.hidden_dim : (k+1)*self.hidden_dim]
#            w2_t = weights2[k*self.input_dim : (k+1)*self.input_dim] 
#            b2_t = biases2[k*self.input_dim : (k+1)*self.input_dim]
#            out = x.matmul(w1_t.t()) + b1_t
#            out = self.non_linearity(out)
#            out = out.matmul(w2_t.t()) + b2_t
            #out = self.non_linearity(out)
            ##--------------#
            
            ##--------------#
            ##Lin et al. '18 model
#            k = int(t/dt)
#            w_t = weights_1[k:(k+1)]  
#            b_t = biases[k:(k+1)]
#            out = x.matmul(w_t.t()) + b_t
#            u_t = weights_2[k:(k+1)]
#            out = self.non_linearity(out)
#            out = out.matmul(u_t)
            ##--------------#

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
                x = x.view(x.size(0), -1)
                aug = torch.zeros(x.shape[0], self.odefunc.augment_dim).to(self.device)
                x_aug = torch.cat([x, aug], 1)
        else:
            x_aug = x

        if self.adjoint:
            ##--------------#
            ##.. Adaptive scheme
            # out = odeint_adjoint(self.odefunc, x_aug, integration_time,
            #                      rtol=self.tol, atol=self.tol, method='dopri5',
            #                      options={'max_num_steps': MAX_NUM_STEPS})
            ##--------------#
            out = odeint_adjoint(self.odefunc, x_aug, integration_time, method='euler', options={'step_size': dt})
        else:
            ##--------------#
            ##.. Adaptive scheme
            # out = odeint(self.odefunc, x_aug, integration_time,
            #              rtol=self.tol, atol=self.tol, method='dopri5',
            #              options={'max_num_steps': MAX_NUM_STEPS})
            ##--------------#
            out = odeint(self.odefunc, x_aug, integration_time, method='euler', options={'step_size': dt})
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
        self.non_linearity = nn.Tanh()

    def forward(self, x, return_features=False):

        features = self.odeblock(x)
        pred = self.linear_layer(features)
        pred = self.non_linearity(pred)

        self.traj = self.odeblock.trajectory(x, time_steps)
        self.proj_traj = self.linear_layer(self.traj)
        self.proj_traj = self.non_linearity(self.proj_traj)

        if return_features:
            return features, pred
        return pred, self.proj_traj
