#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: borjangeshkovski (adapted from https://github.com/EmilienDupont/augmented-neural-odes)
"""
import json
import torch.nn as nn
from numpy import mean
import torch

losses = {'mse': nn.MSELoss(), 
          'cross-entropy': nn.CrossEntropyLoss(), 
          'ell1': nn.SmoothL1Loss()
}

class Trainer():
    """
    Given an optimizer, we write the training loop for minimizing the functional.
    We need several hyperparameters to define the different functionals.

    ***
    -- The boolean "turnpike" indicates whether we integrate the training error over [0,T]
    where T is the time horizon intrinsic to the model.
    -- The boolean "fixed_projector" indicates whether the output layer is given or trained
    -- The float "bound" indicates whether we consider L1+Linfty reg. problem (bound>0.), or 
    L2 reg. problem (bound=0.). If bound>0., then bound represents the upper threshold for the 
    weights+biases.
    ***
    """
    def __init__(self, model, optimizer, device, cross_entropy=True,
                 print_freq=10, record_freq=10, verbose=True, save_dir=None, 
                 turnpike=True, bound=0., fixed_projector=False):
        self.model = model
        self.optimizer = optimizer
        self.cross_entropy = cross_entropy
        self.device = device
        if cross_entropy:
            self.loss_func = losses['cross_entropy']
        else:
            self.loss_func = losses['mse']
        self.print_freq = print_freq
        self.record_freq = record_freq
        self.steps = 0
        self.save_dir = save_dir
        self.verbose = verbose
        self.turnpike = turnpike
        # In case we consider L1-reg. we threshold the norm. 
        # Examples: M \sim T for toy datasets; 200 for mnist
        self.threshold = bound    
        self.fixed_projector = fixed_projector

        self.histories = {'loss_history': [], 'acc_history': [],
                          'epoch_loss_history': [], 'epoch_acc_history': []}
        self.buffer = {'loss': [], 'accuracy': []}
        self.is_resnet = hasattr(self.model, 'num_layers')

    def train(self, data_loader, num_epochs):
        for epoch in range(num_epochs):
            avg_loss = self._train_epoch(data_loader, epoch)
            if self.verbose:
                print("Epoch {}: {:.3f}".format(epoch + 1, avg_loss))

    def _train_epoch(self, data_loader, epoch):
        epoch_loss = 0.
        epoch_acc = 0.
        for i, (x_batch, y_batch) in enumerate(data_loader):
            self.optimizer.zero_grad()
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            if not self.is_resnet:
                y_pred, traj = self.model(x_batch)   
                time_steps = self.model.time_steps 
                T = self.model.T
                dt = T/time_steps
            else:
                # In ResNet, dt=1=T/N_layers.
                y_pred, traj, _ = self.model(x_batch)
                time_steps = self.model.num_layers
                T = time_steps
                dt = 1 

            if not self.turnpike:                                       ## Classical empirical risk minimization
                loss = self.loss_func(y_pred, y_batch)
            else:                                                       ## Augmented empirical risk minimization
                if self.threshold>0:
                    l1_regularization = 0.
                    for param in self.model.parameters():
                        l1_regularization += param.abs().sum()
                    ## lambda = 5*1e-3 for spheres+inside
                    loss = 1.5*sum([self.loss_func(traj[k], y_batch)+self.loss_func(traj[k+1], y_batch) 
                                    for k in range(time_steps-1)]) + 0.005*l1_regularization
                else:
                    if self.fixed_projector: 
                        xd = torch.tensor([[6.0/0.8156, 0.5/(2*0.4525)] if x==1 else [-6.0/0.8156, -2.0/(2*0.4525)] for x in y_batch])
                        loss = self.loss_func(y_pred, y_batch.float())+sum([self.loss_func(traj[k], xd)
                                            +self.loss_func(traj[k+1], xd) for k in range(time_steps-1)])
                    else:
                        ## beta=1.5 for point clouds
                        beta = 1.75                      
                        loss = beta*sum([self.loss_func(traj[k], y_batch)+self.loss_func(traj[k+1], y_batch) 
                                        for k in range(time_steps-1)])
            loss.backward()
            self.optimizer.step()
            
            clipper = WeightClipper(self.threshold)
            if self.threshold>0: 
                self.model.apply(clipper)       # We apply the Linfty constraint to the trained parameters
            
            if self.cross_entropy:
                epoch_loss += self.loss_func(traj[-1], y_batch).item()   
                m = nn.Softmax()
                softpred = m(y_pred)
                softpred = torch.argmax(softpred, 1)  
                epoch_acc += (softpred == y_batch).sum().item()/(y_batch.size(0))       
            else:
                epoch_loss += self.loss_func(y_pred, y_batch).item()
        
            if i % self.print_freq == 0:
                if self.verbose:
                    print("\nEpoch {}/{}".format(i, len(data_loader)))
                    if self.cross_entropy:
                        print("Loss: {:.3f}".format(self.loss_func(traj[-1], y_batch).item()))
                        print("Accuracy: {:.3f}".format((softpred == y_batch).sum().item()/(y_batch.size(0))))
                    else:
                        print("Loss: {:.3f}".format(self.loss_func(y_pred, y_batch).item()))
                        
            self.buffer['loss'].append(self.loss_func(traj[-1], y_batch).item())
            if not self.fixed_projector:
                self.buffer['accuracy'].append((softpred == y_batch).sum().item()/(y_batch.size(0)))

            # At every record_freq iteration, record mean loss and clear buffer
            if self.steps % self.record_freq == 0:
                self.histories['loss_history'].append(mean(self.buffer['loss']))
                if not self.fixed_projector:
                    self.histories['acc_history'].append(mean(self.buffer['accuracy']))

                # Clear buffer
                self.buffer['loss'] = []
                self.buffer['accuracy'] = []

                # Save information in directory
                if self.save_dir is not None:
                    dir, id = self.save_dir
                    with open('{}/losses{}.json'.format(dir, id), 'w') as f:
                        json.dump(self.histories['loss_history'], f)

            self.steps += 1

        # Record epoch mean information
        self.histories['epoch_loss_history'].append(epoch_loss / len(data_loader))
        if not self.fixed_projector:
            self.histories['epoch_acc_history'].append(epoch_acc / len(data_loader))

        return epoch_loss / len(data_loader)

class WeightClipper(object):
    """
    $L^\infty$ constraint, only required if we work with L1-regularization.
    We normalize the weights by dividing by the threshold once the constraint 
    is saturated.
    """
    def __init__(self, threshold, frequency=1):
        self.frequency = frequency
        self.threshold = threshold

    def __call__(self, module):
        if hasattr(module, 'weight') or hasattr(module, 'bias'):
            w = module.weight.data
            b = module.bias.data

            ctrl_norm = w.abs().sum() + b.abs().sum()
            if ctrl_norm > self.threshold:
                w = w*(self.threshold/ctrl_norm)
                b = b*(self.threshold/ctrl_norm)
                module.weight.data = w
                module.bias.data = b
