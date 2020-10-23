#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 14:45:37 2020

@author: borjangeshkovski
"""

import pickle
import torch
from math import sqrt, sin, pi

with open('text.txt', 'rb') as fp:
    out = pickle.load(fp)
    
At = out[0]
bt = out[1]

filtering = 80
dim = len(At[0])


times = torch.linspace(0, 20, 80)

A_times = [0 for j in range(len(times))]
b_times = [0 for j in range(len(times))]


for i, t in enumerate(times):
    for k in range(filtering):
        A_times[i]+= At[k*dim:(k+1)*dim]
        b_times[i]+= bt[k*dim:(k+1)*dim]
        
A_norms = [0 for j in range(len(times))]
b_norms = [0 for j in range(len(times))]

for i, t in enumerate(times):
    A_norms[i] = torch.sum(torch.norm(A_times[i].view(dim,-1), p=2, dim=1)*torch.norm(A_times[i].view(dim,-1), p=2, dim=1))
    b_norms[i] = torch.sum(torch.norm(b_times[i].view(dim, -1), p=2, dim=1)*torch.norm(b_times[i].view(dim, -1), p=2, dim=1))

u_norm = [sqrt(x+y) for x, y in zip(A_norms, b_norms)]

import matplotlib.pyplot as plt

plt.plot(u_norm)
plt.show()