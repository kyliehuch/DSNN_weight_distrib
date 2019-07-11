import torch
import os

state_dict = torch.load("saved_l2.net")
for param_tensor in state_dict:
    if "stdp" not in param_tensor:
        print(param_tensor, "\t", state_dict[param_tensor].size())
        #print(param_tensor, "\t", state_dict[param_tensor])
        #for neuron in state_dict[param_tensor][0]:

conv1_weights = []
for k in range(32):
    cur_weights = []
    for i in range(2):
        for j in range(5):
            for weight in state_dict["conv1.weight"][k,i,j,:]:
                cur_weights.append(weight.item())
    conv1_weights.append(cur_weights)


print("number weights:", len(conv1_weights[0]))
print(conv1_weights[0])


'''
c1_n0_weights = []
for i in range(2):
    for j in range(5):
        for weight in state_dict["conv1.weight"][0,i,j,:]:
            c1_n0_weights.append(weight.item())

print("number weights:", len(c1_n0_weights))
print(c1_n0_weights)
'''
