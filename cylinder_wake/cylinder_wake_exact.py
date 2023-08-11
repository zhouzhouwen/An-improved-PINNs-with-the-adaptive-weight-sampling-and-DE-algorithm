import pandas as pd
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import time
import scipy.io
from scipy.interpolate import interp1d, interp2d
from scipy.interpolate import griddata

# read data
data = scipy.io.loadmat('cylinder_nektar_wake.mat') # N = 5000
U_star = data['U_star']  # N x 2 x T, u and v
P_star = data['p_star']  # N x T, p
t_star = data['t']  # T x 1, t
X_star = data['X_star']  # N x 2, x and y

N = X_star.shape[0] # N = 5000
T = t_star.shape[0] # t = 200

# rearrange data
XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
YY = np.tile(X_star[:, 1:2], (1, T))  # N x T
TT = np.tile(t_star, (1, N)).T  # N x T
UU = U_star[:, 0, :]  # N x T
VV = U_star[:, 1, :]  # N x T
PP = P_star  # N x T

# form a table data
x = XX.flatten()[:, None]  # NT x 1
y = YY.flatten()[:, None]  # NT x 1
t = TT.flatten()[:, None]  # NT x 1
u = UU.flatten()[:, None]  # NT x 1
v = VV.flatten()[:, None]  # NT x 1
p = PP.flatten()[:, None]  # NT x 1
data1 = np.concatenate([x, y, t, u, v, p], 1)
# define domain
data2 = data1[:, :][data1[:, 2] == 4]
data3 = data2[:, :][data2[:, 0] >= 1]
data4 = data3[:, :][data3[:, 0] <= 8]
data5 = data4[:, :][data4[:, 1] >= -2]
data_t4 = data5[:, :][data5[:, 1] <= 2]

# interpolate
x1 = data_t4[:,0].reshape(-1,1)
y1 = data_t4[:,1].reshape(-1,1)
X_star = np.hstack((x1, y1))

x2 = np.linspace(1, 8, 700)
y2 = np.linspace(-2, 2, 400)
x2, y2 = np.meshgrid(x2, y2)

u = griddata(X_star, data_t4[:,3], (x2, y2), method='cubic').reshape(400,700)
v = griddata(X_star, data_t4[:,4], (x2, y2), method='cubic').reshape(400,700)
p = griddata(X_star, data_t4[:,5], (x2, y2), method='cubic').reshape(400,700)

temp = p
temp = np.flipud(temp)
# Create a dataset
df = pd.DataFrame(temp)
print(df)
print(df.shape)


fig = plt.figure(figsize=(7, 4))
plt.title("Result",size=15)
ax = sns.heatmap(data=df,cmap="rainbow",xticklabels=False,yticklabels=False)
plt.show()

