#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: borjangeshkovski
"""

import pickle
from data.dataloaders import ConcentricSphere, Chess, Tricolor, Chess3
import torch

data_dim = 2
num_points_inner = 1000
num_points_outer = 2000

datasets = {"spheres": ConcentricSphere(data_dim, 
                               inner_range=(0., .5), 
                               outer_range=(1., 1.5),
                               num_points_inner=num_points_inner, 
                               num_points_outer=num_points_outer),
			"chess": Chess(3000),
			"tricolor": Tricolor(500, 1000, 2000),
            "chess3": Chess3(3000)
}

_data_line = datasets["chess3"]

# Separate dataset in train and test data.
train_size = int(0.8 * len(_data_line))
test_size = len(_data_line) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(_data_line, 
                                                            [train_size, 
                                                             test_size])

with open('data.txt', 'wb') as fp:
    pickle.dump((train_dataset, test_dataset), fp)