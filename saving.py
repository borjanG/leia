import torch
import pickle
device = torch.device('cpu')
from experiments.dataloaders import *

data_dim = 2
# Examples: 1500, etc.
num_points_lower = 5
num_points_upper = 5

# 1000 vs 2000
num_points_inner = 1000
num_points_outer = 2000

datasets = {
			"spheres": ConcentricSphere(data_dim, inner_range=(0., .5), outer_range=(1., 1.5),
				num_points_inner=num_points_inner, num_points_outer=num_points_outer),
			"checkers": Checkers(num_points_lower, num_points_upper),
			"sines": ShiftedSines(data_dim, shift=1.4, num_points_lower=1500, num_points_upper=1500, noise_scale=0.2)
}

_type = "spheres"
data_line = datasets[_type]

with open('data.txt', 'wb') as fp:
    pickle.dump(data_line, fp)