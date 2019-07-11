import os
import torch
import numpy as np
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

print("number weights, c1n0:", len(c1n0_wts))
print(c1n0_wts)




'''
conv1_weights = []
for k in range(32):
    cur_weights = []
    for i in range(2):
        for j in range(5):
            for weight in state_dict["conv1.weight"][k,i,j,:]:
                cur_weights.append(weight.item())
    conv1_weights.append(cur_weights)

conv2_weights = []
for k in range(150):
    cur_weights = []
    for i in range(32):
        for j in range(2):
            for weight in state_dict["conv1.weight"][k,i,j,:]:
                cur_weights.append(weight.item())
    conv2_weights.append(cur_weights)


print("number weights, c1[0]:", len(conv1_weights[0]))
print(conv1_weights[0])

print("number weights, c1[1]:", len(conv1_weights[1]))
print(conv1_weights[1])
'''
