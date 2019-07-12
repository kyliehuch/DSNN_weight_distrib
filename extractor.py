import torch
import os

f = open("kherweights.txt","w")

state_dict = torch.load("saved_l2.net")
for param_tensor in state_dict:
    if "stdp" not in param_tensor:
        #f.write(param_tensor + "\t" + state_dict[param_tensor].size() + "\n")
        print(param_tensor, "\t", state_dict[param_tensor].size())

f.write("conv1 weights:\n")
for k in range(32):
    cur_weights = []
    for i in range(2):
        for j in range(5):
            for weight in state_dict["conv1.weight"][k,i,j,:]:
                cur_weights.append(weight.item())
    f.write("c1n{}_wts = {}\n".format(k, cur_weights))

f.write("\nconv1 weights:\n")
for k in range(150):
    cur_weights = []
    for i in range(32):
        for j in range(2):
            for weight in state_dict["conv1.weight"][k,i,j,:]:
                cur_weights.append(weight.item())
    f.write("c2n{}_wts = {}\n".format(k, cur_weights))

f.close()
