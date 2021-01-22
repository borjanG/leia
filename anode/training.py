import json
import torch.nn as nn
from numpy import mean
import torch

##--------------#
T = 15.0
time_steps = 15
dt = T/time_steps
threshold = 5
##--------------#

class WeightClipper(object):

    def __init__(self, frequency=1):
        self.frequency = frequency

    def __call__(self, module):
        
        if hasattr(module, 'weight') or hasattr(module, 'bias'):
            w = module.weight.data
            b = module.bias.data
            ctrl_norm = w.abs().sum() + b.abs().sum()
            if ctrl_norm > threshold:
                w = w*(threshold/ctrl_norm)
                b = b*(threshold/ctrl_norm)
                #w = torch.clamp(w, -threshold, threshold)
                #b = torch.clamp(b, -threshold, threshold)
                module.weight.data = w
                module.bias.data = b

class Trainer():
    def __init__(self, model, optimizer, device, classification=True,
                 print_freq=10, record_freq=10, verbose=True, save_dir=None):
        self.model = model
        self.optimizer = optimizer
        self.classification = classification
        self.device = device
        self.turnpike = True
        if self.classification:
            self.loss_func = nn.MSELoss()
            #self.loss_func = nn.SmoothL1Loss()
            #self.loss_func = nn.CrossEntropyLoss()
        else:
            #self.loss_func = nn.SmoothL1Loss()
            self.loss_func = nn.MSELoss()
        self.print_freq = print_freq
        self.record_freq = record_freq
        self.steps = 0
        self.save_dir = save_dir
        self.verbose = verbose
        
        self.histories = {'loss_history': [], 'epoch_loss_history': []}
        self.buffer = {'loss': []}
        self.is_resnet = hasattr(self.model, 'num_layers')

    def train(self, data_loader, num_epochs):
        for epoch in range(num_epochs):
            avg_loss = self._train_epoch(data_loader)
            if self.verbose:
                print("Epoch {}: {:.3f}".format(epoch + 1, avg_loss))

    def _train_epoch(self, data_loader):
        epoch_loss = 0.
        for i, (x_batch, y_batch) in enumerate(data_loader):
            self.optimizer.zero_grad()

            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            
            if not self.is_resnet:
                y_pred, traj = self.model(x_batch)   
            else:
                y_pred, traj, _ = self.model(x_batch)
            
            ell1 = True
            
            if not self.turnpike:
                loss = self.loss_func(y_pred, y_batch.float())
            else:
                import torch
                if ell1:
                    l1_regularization = 0.
                    for param in self.model.parameters():
                        l1_regularization += param.abs().sum()
                    loss = 1.5*sum([self.loss_func(traj[k], y_batch.float())+self.loss_func(traj[k+1], y_batch.float()) for k in range(time_steps-1)]) + 0.01*l1_regularization
                    #loss = 1.5*sum([self.loss_func(traj[k], y_batch.float())+self.loss_func(traj[k+1], y_batch.float()) for k in range(time_steps-1)]) + dt*0.01*l1_regularization
                else:
                    xd = torch.tensor([[2.0,2.0] if x==1 else [-2.0,-2.0] for x in y_batch])
                    ## TURNPIKE 1
                    loss = 0.1*self.loss_func(y_pred, y_batch.float())+2.5*dt*sum([self.loss_func(traj[k], xd)+self.loss_func(traj[k+1], xd) for k in range(time_steps-1)])
            
                    ## TURNPIKE 2 (0.1*dt/2 before)
                    #loss = 1.5*sum([self.loss_func(traj[k], y_batch)+self.loss_func(traj[k+1], y_batch) for k in range(time_steps-1)])
                    #loss = 1.5*sum([self.loss_func(traj[k], y_batch.float())+self.loss_func(traj[k+1], y_batch.float()) for k in range(time_steps-1)])
                
            loss.backward()
            self.optimizer.step()
            
            #Contrainte Linfty < M
            #pars = [self.model.state_dict()[_] for _ in self.model.state_dict()]
            #for param in self.model.parameters():
            #    _norm = param.abs().sum()
            #    if _norm > threshold:
            #print('hi', self.model.)
            
            #if pars[0] or pars[1]
                
            
            clipper = WeightClipper()
            self.model.apply(clipper)
            epoch_loss += self.loss_func(traj[-1], y_batch).item()
            
            if i % self.print_freq == 0:
                if self.verbose:
                    print("\nEpoch {}/{}".format(i, len(data_loader)))
                    print("Loss: {:.3f}".format(self.loss_func(traj[-1], y_batch).item()))
                        
            self.buffer['loss'].append(loss.item())
            if self.steps % self.record_freq == 0:
                self.histories['loss_history'].append(mean(self.buffer['loss']))
                self.buffer['loss'] = []
                # Save information in directory
                if self.save_dir is not None:
                    dir, id = self.save_dir
                    with open('{}/losses{}.json'.format(dir, id), 'w') as f:
                        json.dump(self.histories['loss_history'], f)
            self.steps += 1

        # Record epoch mean information
        self.histories['epoch_loss_history'].append(epoch_loss / len(data_loader))

        return epoch_loss / len(data_loader)
