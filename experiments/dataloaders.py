import glob
import imageio
import numpy as np
import torch
from math import pi
import random as rand
from random import random
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Normal
from torchvision import datasets, transforms

class Data1D(Dataset):
    def __init__(self, num_points, target_flip=False, noise_scale=0.0):
        self.num_points = num_points
        self.target_flip = target_flip
        self.noise_scale = noise_scale
        self.data = []
        self.targets = []

        noise = Normal(loc=0., scale=self.noise_scale)

        for _ in range(num_points):
            if random() > 0.5:
                data_point = 1.0
                target = 1.0
            else:
                data_point = -1.0
                target = -1.0

            if self.target_flip:
                target *= -1

            if self.noise_scale > 0.0:
                data_point += noise.sample()

            self.data.append(torch.Tensor([data_point]))
            self.targets.append(torch.Tensor([target]))

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.num_points


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
            #__ = torch.tensor(0)
            #__ = __.type(torch.long)
            #self.targets.append(__)
            self.targets.append(torch.Tensor([-1]))

        for _ in range(self.num_points_outer):
            self.data.append(
                random_point_in_sphere(dim, outer_range[0], outer_range[1])
            )
            #__ = torch.tensor(1)
            #__ = __.type(torch.long)
            #self.targets.append(__)
            self.targets.append(torch.Tensor([1]))

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
            self.targets.append(torch.Tensor([-1]))

        for _ in range(self.num_points//4):
            self.data.append(random_point_in_square(0.1, 0.1, 1.4, 1.4))
            self.targets.append(torch.Tensor([-1]))

        for _ in range(self.num_points//4):
            self.data.append(random_point_in_square(-1.4, 0.1, -0.1, 1.4))
            self.targets.append(torch.Tensor([1]))

        for _ in range(self.num_points//4):
            self.data.append(random_point_in_square(0.1, -1.4, 1.4, -0.1))
            self.targets.append(torch.Tensor([1]))

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
            #self.targets.append(torch.Tensor([-1]))

        for _ in range(self.num_points_r):
            self.data.append(
                random_point_in_sphere(2, 0.75, 1.25)
            )
            __ = torch.tensor(1)
            __ = __.type(torch.long)
            self.targets.append(__)
            #self.targets.append(torch.Tensor([1]))

        for _ in range(self.num_points_g):
            self.data.append(
                random_point_in_sphere(2, 1.5, 2.0)
            )
            __ = torch.tensor(2)
            __ = __.type(torch.long)
            self.targets.append(__)
            #self.targets.append(torch.Tensor([1,0]))

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


def mnist(batch_size=64, size=28, path_to_data='../../mnist_data'):
    
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
