# Extract sparcity for all neurons, plot histogram for each layer
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

NUM_C1INPS = 50
NUM_C2INPS = 128
state_dict = torch.load("saved_l2.net")

c1wts = []
for k in range(32):
    cur_wts = []
    for i in range(2):
        for j in range(5):
            for wt in state_dict["conv1.weight"][k,i,j,:]:
                cur_wts.append(wt.item())
    c1wts.append(cur_wts)

c2wts = []
for k in range(150):
    cur_wts = []
    for i in range(32):
        for j in range(2):
            for wt in state_dict["conv2.weight"][k,i,j,:]:
                cur_wts.append(wt.item())
    c2wts.append(cur_wts)

c1_act = []    # number of inputs with weights close to 1 for each neuron
for nrn in c1wts:
    wt_cnt = 0
    for wt in nrn:
        if wt >= 0.95:
            wt_cnt +=
    c1_act.append(wt_cnt)

c2_act = []    # number of inputs with weights close to 1 for each neuron
for nrn in c2wts:
    wt_cnt = 0
    for wt in nrn:
        if wt >= 0.95:
            wt_cnt +=
    c2_act.append(wt_cnt)

c1_spars = []
for act in c1_act:
    c1_spars.append(act/NUM_C1INPS)

c2_spars = []
for act in c2_act:
    c2_spars.append(act/NUM_C2INPS)

print("Convolution layer 1:\n")
print("conv1 activation: {}\n".format(c1_act))
print("conv1 sparsity: {}\n\n".format(c1_spars))
print("Convolution layer 2:\n")
print("conv2 activation: {}\n".format(c2_act))
print("conv2 sparsity: {}\n\n".format(c1_spars))
