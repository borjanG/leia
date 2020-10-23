import json
import torch
import torch.nn as nn
from numpy import mean

##--------------#
##.. Turnpike
T = 45.0
time_steps = 180
dt = T/time_steps

##.. Not Turnpike
#T = 81.0                
#time_steps = int(pow(T, 1.5))
#dt = T/pow(T, 1.5)
##--------------#


class Trainer():
    def __init__(self, model, optimizer, device, classification=True,
                 print_freq=10, record_freq=10, verbose=True, save_dir=None):
        self.model = model
        self.optimizer = optimizer
        self.classification = classification
        self.device = device
        if self.classification:
            self.loss_func = nn.MSELoss()
        else:
            #self.loss_func = nn.SmoothL1Loss()
            self.loss_func = nn.MSELoss()
        self.print_freq = print_freq
        self.record_freq = record_freq
        self.steps = 0
        self.save_dir = save_dir
        self.verbose = verbose

        self.histories = {'loss_history': [], 'nfe_history': [],
                          'bnfe_history': [], 'total_nfe_history': [],
                          'epoch_loss_history': [], 'epoch_nfe_history': [],
                          'epoch_bnfe_history': [], 'epoch_total_nfe_history': []}
        self.buffer = {'loss': [], 'nfe': [], 'bnfe': [], 'total_nfe': []}

        self.is_resnet = hasattr(self.model, 'num_layers')

    def train(self, data_loader, num_epochs):
        for epoch in range(num_epochs):
            avg_loss = self._train_epoch(data_loader)
            if self.verbose:
                print("Epoch {}: {:.3f}".format(epoch + 1, avg_loss))

    def _train_epoch(self, data_loader):
        epoch_loss = 0.
        epoch_nfes = 0
        epoch_backward_nfes = 0
        for i, (x_batch, y_batch) in enumerate(data_loader):
            self.optimizer.zero_grad()

            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            #xd = torch.tensor([[2.0,2.0] if x==1 else [-2.0,-2.0] for x in y_batch])            
            #xd = torch.tensor([[2.0,2.0,2.0] if x==1 else [-2.0,-2.0,-2.0] for x in y_batch])
 
            y_pred, traj = self.model(x_batch)

            if not self.is_resnet:
                iteration_nfes = self._get_and_reset_nfes()
                epoch_nfes += iteration_nfes            

            ## ASYMPTOTICS
            #loss = self.loss_func(y_pred, y_batch)

            ## TURNPIKE 1
            #loss = 0.1*self.loss_func(y_pred, y_batch)+100*dt/2*sum([self.loss_func(traj[k], xd)+self.loss_func(traj[k+1], xd) for k in range(time_steps-1)])
            
            ## TURNPIKE 2 (0.1*dt/2 before)
            loss = 1.5*sum([self.loss_func(traj[k], y_batch)+self.loss_func(traj[k+1], y_batch) for k in range(time_steps-1)])

            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

            if not self.is_resnet:
                iteration_backward_nfes = self._get_and_reset_nfes()
                epoch_backward_nfes += iteration_backward_nfes

            if i % self.print_freq == 0:
                if self.verbose:
                    print("\nIteration {}/{}".format(i, len(data_loader)))
                    print("Loss: {:.3f}".format(loss.item()))
                    if not self.is_resnet:
                        print("NFE: {}".format(iteration_nfes))
                        print("BNFE: {}".format(iteration_backward_nfes))
                        print("Total NFE: {}".format(iteration_nfes + iteration_backward_nfes))

            self.buffer['loss'].append(loss.item())
            if not self.is_resnet:
                self.buffer['nfe'].append(iteration_nfes)
                self.buffer['bnfe'].append(iteration_backward_nfes)
                self.buffer['total_nfe'].append(iteration_nfes + iteration_backward_nfes)

            if self.steps % self.record_freq == 0:
                self.histories['loss_history'].append(mean(self.buffer['loss']))

                if not self.is_resnet:
                    self.histories['nfe_history'].append(mean(self.buffer['nfe']))
                    self.histories['bnfe_history'].append(mean(self.buffer['bnfe']))
                    self.histories['total_nfe_history'].append(mean(self.buffer['total_nfe']))

                self.buffer['loss'] = []
                self.buffer['nfe'] = []
                self.buffer['bnfe'] = []
                self.buffer['total_nfe'] = []

                if self.save_dir is not None:
                    dir, id = self.save_dir
                    with open('{}/losses{}.json'.format(dir, id), 'w') as f:
                        json.dump(self.histories['loss_history'], f)
                    if not self.is_resnet:
                        with open('{}/nfes{}.json'.format(dir, id), 'w') as f:
                            json.dump(self.histories['nfe_history'], f)
                        with open('{}/bnfes{}.json'.format(dir, id), 'w') as f:
                            json.dump(self.histories['bnfe_history'], f)
                        with open('{}/total_nfes{}.json'.format(dir, id), 'w') as f:
                            json.dump(self.histories['total_nfe_history'], f)

            self.steps += 1

        self.histories['epoch_loss_history'].append(epoch_loss / len(data_loader))
        if not self.is_resnet:
            self.histories['epoch_nfe_history'].append(float(epoch_nfes) / len(data_loader))
            self.histories['epoch_bnfe_history'].append(float(epoch_backward_nfes) / len(data_loader))
            self.histories['epoch_total_nfe_history'].append(float(epoch_backward_nfes + epoch_nfes) / len(data_loader))

        return epoch_loss / len(data_loader)

    def _get_and_reset_nfes(self):
        if hasattr(self.model, 'odeblock'):  # If we are using ODENet
            iteration_nfes = self.model.odeblock.odefunc.nfe
            self.model.odeblock.odefunc.nfe = 0
        else: 
            iteration_nfes = self.model.odefunc.nfe
            self.model.odefunc.nfe = 0
        return iteration_nfes
