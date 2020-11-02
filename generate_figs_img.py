#import torch
#device = torch.device('cpu')
#from torch.utils.data import DataLoader
from viz.plots import plt_state_component
#from anode.training import Trainer
#from anode.models import ODENet
#from experiments.dataloaders import Mnist1d


#data = Mnist1d()
#dataloader = DataLoader(data, batch_size=256, shuffle=True)
#
#for inputs, targets in dataloader:
#    break

#from experiments.dataloaders import mnist
# 
#dataloader, test_loader = mnist(256)
#output_dim = 10
#
#for inputs, targets in dataloader:
#    break
#
#hidden_dim = pow(28,2)+32
#data_dim = pow(28,2)
#anode = ODENet(device, data_dim, hidden_dim, augment_dim=32, non_linearity='tanh')
#
#T = 5.0
#num_steps = 5
#dt = T/num_steps
#
#optimizer_anode = torch.optim.Adam(anode.parameters(), lr=1e-3, weight_decay=0.1)
#
#
#trainer_anode = Trainer(anode, optimizer_anode, device)
#num_epochs = 20
#trainer_anode.train(dataloader, num_epochs)
#plt_state_component(anode, inputs.view(inputs.size(0),-1), targets, timesteps=num_steps, component=150, save_fig='first.pdf')

################################

# =============================================================================
import torch
device = torch.device('cpu')
from anode.discrete_models import ResNet
from anode.training import Trainer
from experiments.dataloaders import mnist
 
data_loader, test_loader = mnist(256)
img_size = (1, 28, 28)
output_dim = 10
 
model = ResNet(pow(28,2), 32, 40, output_dim=10, is_img=True)
 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01)
 
trainer = Trainer(model, optimizer, device, classification=True)
num_epochs = 10
trainer.train(data_loader, num_epochs)

for inputs, targets in data_loader:
    break

plt_state_component(model, inputs.view(inputs.size(0),-1), targets, timesteps=40, component=397, save_fig='first.pdf')

# =============================================================================
