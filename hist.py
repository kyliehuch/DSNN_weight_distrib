# Code to produce histogram of weights from array
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

state_dict = torch.load("saved_l2.net")
for param_tensor in state_dict:
    if "stdp" not in param_tensor:
        print(param_tensor, "\t", state_dict[param_tensor].size())

c1n0_wts = []
for i in range(2):
    for j in range(5):
        for wt in state_dict["conv1.weight"][0,i,j,:]:
            c1n0_wts.append(wt.item())

wt_hist = np.histogram(c1n0_wts, bins=10, range=(0.0, 1.0))
plt.hist(wt_hist)
plt.title("Histogram of single c1 neuron's weight distribution")
plt.show()
