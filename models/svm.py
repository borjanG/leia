#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 14:12:39 2021

@author: borjangeshkovski
"""

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
from matplotlib import rc

# we create 40 separable points
X, y = make_blobs(n_samples=150, centers=2, random_state=6)
color = ['crimson' if y[i] > 0.0 else 'dodgerblue' for i in range(len(y))]

# fit the model, don't regularize for illustration purposes
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X, y)

plt.scatter(X[:, 0], 
            X[:, 1], 
            c=color, 
            s=30, 
            cmap=plt.cm.Paired, 
            marker = 'o', 
            linewidth=0.45, 
            edgecolors='black', 
            alpha=0.85)

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

ax.set_axisbelow(True)
ax.xaxis.grid(color='lightgray', linestyle='dotted')
ax.yaxis.grid(color='lightgray', linestyle='dotted')

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)


#rc("text", usetex = True)
#font = {'size'   : 13}
#rc('font', **font)
ax.set_facecolor('whitesmoke')
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
plt.title('Margin and max-margin hyperplane', fontsize=11)
#plt.xlabel(r'x_1')
#plt.xlabel(r'x_2')

#plt.rc('grid', linestyle="dotted", color='lightgray')

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.65,
           linestyles=['--', '-', '--'])
# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')

plt.savefig('svm.pdf', format='pdf', bbox_inches='tight')
plt.show()
