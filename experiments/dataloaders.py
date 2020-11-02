import glob
import imageio
import numpy as np
import torch
from math import pi
from random import random
#import random
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
            self.targets.append(torch.Tensor([-1]))

        for _ in range(self.num_points_outer):
            self.data.append(
                random_point_in_sphere(dim, outer_range[0], outer_range[1])
            )
            self.targets.append(torch.Tensor([1]))

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)
    
class Mnist1d(Dataset):
    def __init__(self):
    
        self.data = []
        self.targets = []
        
        data_loader, test_loader = mnist(batch_size=256, size=28)

        for _inputs, targets in data_loader:
            break

        inputs = torch.zeros(256, pow(28, 2))
        for i, x in enumerate(_inputs):
            inputs[i] = x.reshape(-1)

        for _ in range(256):
            self.data.append(inputs[_])
            self.targets.append(targets[_])
            #if targets[_]>5:
            #    self.targets.append(torch.Tensor([1]))
            #else:
            #    self.targets.append(torch.Tensor([-1]))

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)


class ShiftedSines(Dataset):
    def __init__(self, dim, shift, num_points_upper, num_points_lower,
                 noise_scale):
        self.dim = dim
        self.shift = shift
        self.num_points_upper = num_points_upper
        self.num_points_lower = num_points_lower
        self.noise_scale = noise_scale

        noise = Normal(loc=0., scale=self.noise_scale)

        self.data = []
        self.targets = []

        for i in range(self.num_points_upper + self.num_points_lower):
            if i < self.num_points_upper:
                label = 1
                y_shift = shift / 2.
            else:
                label = -1
                y_shift = - shift / 2.

            x = 2 * torch.rand(1) - 1 
            y = torch.sin(pi * x) + noise.sample() + y_shift

            if self.dim == 1:
                self.data.append(torch.Tensor([y]))
            elif self.dim == 2:
                self.data.append(torch.cat([x, y]))
            else:
                random_higher_dims = 2 * torch.rand(self.dim - 2) - 1
                self.data.append(torch.cat([x, y, random_higher_dims]))

            self.targets.append(torch.Tensor([label]))

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)


def random_point_in_sphere(dim, min_radius, max_radius):
    
    unif = random()
    distance = (max_radius - min_radius) * (unif ** (1. / dim)) + min_radius
    direction = torch.randn(dim)
    unit_direction = direction / torch.norm(direction, 2)
    return distance * unit_direction

class Checkers(Dataset):
    def __init__(self, num_points_upper, num_points_lower):
        
        self.num_points_upper = num_points_upper
        self.num_points_lower = num_points_lower
        self.data = []
        self.targets = []

        for i in range(self.num_points_upper + self.num_points_lower):
            if i < self.num_points_upper/4:
                label = 1
                y = 0.5 * torch.rand(1) - 1  # Random point between -1 and 1
            elif self.num_points_upper/4 <= i < self.num_points_upper/2:
                label = -1
                y = 0.5 * torch.rand(1) - 0.5
            elif self.num_points_upper/2 <= i < 3*self.num_points_upper/4:
                label = 1
                y = 0.5 * torch.rand(1) 
            else:
                label = -1
                y = 0.5 * torch.rand(1) + 0.5
                
            self.data.append(torch.Tensor([y]))
            self.targets.append(torch.Tensor([label]))
        
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


def cifar10(batch_size=64, size=32, path_to_data='../../cifar10_data'):
    
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


def tiny_imagenet(batch_size=64, path_to_data='../../tiny-imagenet-200/'):
    
    imagenet_data = TinyImageNet(root_folder=path_to_data,
                                 transform=transforms.ToTensor())
    imagenet_loader = DataLoader(imagenet_data, batch_size=batch_size,
                                 shuffle=True)
    return imagenet_loader


class TinyImageNet(Dataset):
    
    def __init__(self, root_folder='../../tiny-imagenet-200/', transform=None):
        self.root_folder = root_folder
        self.transform = transform
        self.imgs_and_classes = [] 

        train_folder = root_folder + 'train/'
        class_folders = glob.glob(train_folder + '*')

        for i, class_folder in enumerate(class_folders):
            image_paths = glob.glob(class_folder + '/images/*.JPEG')
            for image_path in image_paths:
                self.imgs_and_classes.append((image_path, i))

        self.transform = transform

    def __len__(self):
        return len(self.imgs_and_classes)

    def __getitem__(self, idx):
        img_path, label = self.imgs_and_classes[idx]
        img = imageio.imread(img_path)

        if self.transform:
            img = self.transform(img)

        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)

        return img, label
