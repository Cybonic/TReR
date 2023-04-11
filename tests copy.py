# Importing libraries

import numpy as np
import pandas as pd
import scipy as stats


import torch
import torch.nn as nn
import torch.nn.functional as F

def compt_y_table(v):
	n = v.shape[0]
	table = torch.zeros((n,n))
	for i in range(n):
		for j in range(n):
			if y[i]>y[j]:
				table[i,j] = 1
	return table

def list_loss(p,table):
	loss = 0
	for i in range(n):
		for j in range(n):
			if table[i,j]:
				loss = loss + torch.log2(1+torch.exp(-(p[i]-p[j])))
	return loss

y = torch.tensor([0.5,0.4,0.3])
py = F.softmax(y).unsqueeze(0)
print(py)



p =  torch.tensor([0.3,0.4,0.3])
p =  torch.tensor([1.2,2.3,3.0])
pz = F.softmax(p).unsqueeze(-1)

loss = -torch.matmul(py,torch.log(pz))

table = compt_y_table(y)

print(table)
n = y.shape[0]

des = np.random.random((1,3))
print(des)
y =  np.random.randint(0,2,(1,3,3))
print(y)
combo_idx = np.arange(3)
permute = np.array([np.array([a, b]) for idx, a in enumerate(combo_idx) for b in combo_idx[idx + 1:]])

print(permute)
sub = des[0,permute[:,0]] - des[0,permute[:,1]] 
print(des[0])
print(sub)
permlabel = y[0][permute[:,0],permute[:,1]]
print(permlabel)

f = permlabel*sub

print(f)