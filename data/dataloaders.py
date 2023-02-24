#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: borjangeshkovski
"""

import numpy as np
import torch
import random as rand
from random import random
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Normal
from torchvision import datasets, transforms

class ConcentricSphere(Dataset):
    def __init__(self, dim, inner_range, outer_range, num_points_inner,
                 num_points_outer):
        self.dim = dim
        self.inner_range = inner_range
        self.outer_range = outer_range
        self.num_points_inner = num_points_inner
        self.num_points_outer = num_points_outer

        self.data = []
        self.targets = []

        # Generate data for inner sphere
        for _ in range(self.num_points_inner):
            self.data.append(
                random_point_in_sphere(dim, inner_range[0], inner_range[1])
            )
            # Cross-entropy
            __ = torch.tensor(0)
            __ = __.type(torch.long)
            self.targets.append(__)
            
            # MSE + sigmoid
            #self.targets.append(torch.Tensor([-1]))

        for _ in range(self.num_points_outer):
            self.data.append(
                random_point_in_sphere(dim, outer_range[0], outer_range[1])
            )
            # Cross entropy
            __ = torch.tensor(1)
            __ = __.type(torch.long)
            self.targets.append(__)
            
            # MSE + sigmoid
            #self.targets.append(torch.Tensor([1]))

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)
    
class Chess(Dataset):
    def __init__(self, num_points):
        self.num_points = num_points

        self.data = []
        self.targets = []

        for _ in range(self.num_points//4):
            self.data.append(random_point_in_square(-1.4, -1.4, -0.1, -0.1))
            ## MSE + sigmoid
            #self.targets.append(torch.Tensor([-1]))
            
            ## Cross-entropy
            __ = torch.tensor(0)
            __ = __.type(torch.long)
            self.targets.append(__)

        for _ in range(self.num_points//4):
            self.data.append(random_point_in_square(0.1, 0.1, 1.4, 1.4))
            ## MSE + sigmoid
            #self.targets.append(torch.Tensor([-1]))
            
            ## Cross-entropy
            __ = torch.tensor(0)
            __ = __.type(torch.long)
            self.targets.append(__)

        for _ in range(self.num_points//4):
            self.data.append(random_point_in_square(-1.4, 0.1, -0.1, 1.4))
            ## MSE
            #self.targets.append(torch.Tensor([1]))
            
            ## Cross-entropy
            __ = torch.tensor(1)
            __ = __.type(torch.long)
            self.targets.append(__)

        for _ in range(self.num_points//4):
            self.data.append(random_point_in_square(0.1, -1.4, 1.4, -0.1))
            ## MSE
            #self.targets.append(torch.Tensor([1]))
            
            ## Cross-entropy
            __ = torch.tensor(1)
            __ = __.type(torch.long)
            self.targets.append(__)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)
    
    
class Chess3(Dataset):
    def __init__(self, num_points):
        self.num_points = num_points

        self.data = []
        self.targets = []

        for _ in range(self.num_points//9):
            self.data.append(random_point_in_square(-0.9, -0.9, -0.37, -0.37))
            ## MSE + sigmoid
            self.targets.append(torch.Tensor([-1]))
            ## Cross-entropy
            #__ = torch.tensor(0)
            #__ = __.type(torch.long)
            #self.targets.append(__)

        for _ in range(self.num_points//9):
            self.data.append(random_point_in_square(-0.28, -0.9, 0.28, -0.37))
            ## MSE + sigmoid
            self.targets.append(torch.Tensor([1]))
            ## Cross-entropy
            #__ = torch.tensor(1)
            #__ = __.type(torch.long)
            #self.targets.append(__)

        for _ in range(self.num_points//9):
            self.data.append(random_point_in_square(0.37, -0.9, 0.9, -0.37))
            ## MSE
            self.targets.append(torch.Tensor([-1]))
            ## Cross-entropy
            #__ = torch.tensor(0)
            #__ = __.type(torch.long)
            #self.targets.append(__)

        for _ in range(self.num_points//9):
            self.data.append(random_point_in_square(-0.9, -0.28, -0.35, 0.28))
            ## MSE
            self.targets.append(torch.Tensor([1]))
            ## Cross-entropy
            #__ = torch.tensor(1)
            #__ = __.type(torch.long)
            #self.targets.append(__)
            
        for _ in range(self.num_points//9):
            self.data.append(random_point_in_square(-0.28, -0.28, 0.28, 0.28))
            ## MSE
            self.targets.append(torch.Tensor([-1]))
            ## Cross-entropy
            #__ = torch.tensor(0)
            #__ = __.type(torch.long)
            #self.targets.append(__)
            
        for _ in range(self.num_points//9):
            self.data.append(random_point_in_square(0.37, -0.28, 0.9, 0.28))
            ## MSE
            self.targets.append(torch.Tensor([1]))
            
            ## Cross-entropy
            #__ = torch.tensor(1)
            #__ = __.type(torch.long)
            #self.targets.append(__)
            
        for _ in range(self.num_points//9):
            self.data.append(random_point_in_square(-0.9, 0.37, -0.37, 0.9))
            ## MSE
            self.targets.append(torch.Tensor([-1]))
            
            ## Cross-entropy
            #__ = torch.tensor(0)
            #__ = __.type(torch.long)
            #self.targets.append(__)
            
        for _ in range(self.num_points//9):
            self.data.append(random_point_in_square(-0.28, 0.37, 0.28, 0.9))
            ## MSE
            self.targets.append(torch.Tensor([1]))
            
            ## Cross-entropy
            #__ = torch.tensor(1)
            #__ = __.type(torch.long)
            #self.targets.append(__)
            
        for _ in range(self.num_points//9):
            self.data.append(random_point_in_square(0.37, 0.37, 0.9, 0.9))
            ## MSE
            self.targets.append(torch.Tensor([-1]))
            
            ## Cross-entropy
            #__ = torch.tensor(0)
            #__ = __.type(torch.long)
            #self.targets.append(__)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)
        
def random_point_in_square(x1, y1, x2, y2):
    return torch.tensor([rand.uniform(x1, x2), rand.uniform(y1, y2)])

def random_point_in_sphere(dim, min_radius, max_radius):
    unif = random()
    distance = (max_radius - min_radius) * (unif ** (1. / dim)) + min_radius
    direction = torch.randn(dim)
    unit_direction = direction / torch.norm(direction, 2)
    scale = 0.15
    if scale > 0.0:
        noise = Normal(loc=0., scale=scale)
        return distance * unit_direction + noise.sample() 
    return distance * unit_direction

class Tricolor(Dataset):
    def __init__(self, num_points_b, num_points_g, num_points_r):

        self.num_points_b = num_points_b
        self.num_points_g = num_points_g
        self.num_points_r = num_points_r

        self.data = []
        self.targets = []

        # Generate data for inner sphere
        for _ in range(self.num_points_b):
            self.data.append(
                random_point_in_sphere(2, 0.0, 0.5)
            )
            __ = torch.tensor(0)
            __ = __.type(torch.long)
            self.targets.append(__)
            ## MSE
            #self.targets.append(torch.Tensor([1, 1]))

        for _ in range(self.num_points_r):
            self.data.append(
                random_point_in_sphere(2, 0.75, 1.25)
            )
            __ = torch.tensor(1)
            __ = __.type(torch.long)
            self.targets.append(__)
            ## MSE
            #self.targets.append(torch.Tensor([1, -1]))

        for _ in range(self.num_points_g):
            self.data.append(
                random_point_in_sphere(2, 1.5, 2.0)
            )
            __ = torch.tensor(2)
            __ = __.type(torch.long)
            self.targets.append(__)
            ## MSE
            #self.targets.append(torch.Tensor([-1, -1]))

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)

def dataset_to_numpy(dataset):
    num_points = len(dataset)
    X = np.zeros((num_points, dataset.dim))
    y = np.zeros((num_points, 1))
    for i in range(num_points):
        X[i] = dataset.data[i].numpy()
        y[i] = dataset.targets[i].item()
    return X.astype('float32'), y.astype('float32')

def mnist(batch_size=64, 
            size=28, 
            path_to_data='../../mnist_data'):
    all_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    train_data = datasets.MNIST(path_to_data, train=True, download=True,
                                transform=all_transforms)
    test_data = datasets.MNIST(path_to_data, train=False,
                               transform=all_transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

def cifar10(batch_size=64, 
            size=32, 
            path_to_data='../../cifar10_data'):
    all_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    train_data = datasets.CIFAR10(path_to_data, train=True, download=True,
                                  transform=all_transforms)
    test_data = datasets.CIFAR10(path_to_data, train=False,
                                 transform=all_transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

def fashion_mnist(batch_size=64, 
                    size=28, 
                    path_to_data='../../fashion-mnist_data'):
    all_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    train_data = datasets.FashionMNIST(path_to_data, train=True, download=True,
                                  transform=all_transforms)
    test_data = datasets.FashionMNIST(path_to_data, train=False,
                                 transform=all_transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader
