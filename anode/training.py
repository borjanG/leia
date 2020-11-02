import torch.nn as nn

##--------------#
##.. Turnpike
T = 40.0
time_steps = 40
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
            #self.loss_func = nn.MSELoss()
            self.loss_func = nn.CrossEntropyLoss()
        else:
            #self.loss_func = nn.SmoothL1Loss()
            self.loss_func = nn.MSELoss()
        self.print_freq = print_freq
        self.record_freq = record_freq
        self.steps = 0
        self.save_dir = save_dir
        self.verbose = verbose

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
            
            y_pred, traj, _ = self.model(x_batch)
            #y_pred = self.model(x_batch)

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
                    print("\nEpoch {}/{}".format(i, len(data_loader)))
                    #print("Loss: {:.3f}".format(loss.item()))
                    print("Loss: {:.3f}".format(self.loss_func(y_pred, y_batch).item()))
                    if not self.is_resnet:
                        print("NFE: {}".format(iteration_nfes))
                        print("Total NFE: {}".format(iteration_nfes + iteration_backward_nfes))
            
            self.steps += 1

        return epoch_loss / len(data_loader)

    def _get_and_reset_nfes(self):
        if hasattr(self.model, 'odeblock'):  # If we are using ODENet
            iteration_nfes = self.model.odeblock.odefunc.nfe
            self.model.odeblock.odefunc.nfe = 0
        else: 
            iteration_nfes = self.model.odefunc.nfe
            self.model.odefunc.nfe = 0
        return iteration_nfes
