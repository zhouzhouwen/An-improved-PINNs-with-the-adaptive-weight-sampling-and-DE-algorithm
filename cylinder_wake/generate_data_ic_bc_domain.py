# -*- coding: utf-8 -*-
"""MyProblem.py"""
import matplotlib.pyplot as plt
import numpy as np
import scipy

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
data2 = data1[:, :][data1[:, 2] <= 1.901]
data3 = data2[:, :][data2[:, 0] >= 1]
data4 = data3[:, :][data3[:, 0] <= 8]
data5 = data4[:, :][data4[:, 1] >= -2]
data_domain = data5[:, :][data5[:, 1] <= 2]
print(data_domain.shape)

# BC
# x_bc指的是流域的四条边，其他同理
# PS: y is the edge of x
data_x1 = data_domain[:, :][data_domain[:, 1] == -2]
data_x2 = data_domain[:, :][data_domain[:, 1] == 2]
data_y1 = data_domain[:, :][data_domain[:, 0] == 1]
data_y8 = data_domain[:, :][data_domain[:, 0] == 8]
data_sup_b_train = np.concatenate([data_y1, data_y8, data_x1, data_x2], 0)
print(data_sup_b_train.shape)

# IC
data_t0 = data_domain[:, :][data_domain[:, 2] == 0] # t == 0
print(data_t0.shape)

np.save("data_bc",data_sup_b_train)
np.save("data_ic",data_t0)
np.save("data_domain",data_domain)

plt.scatter(data_domain[data_domain[:, 2] == 0.1][:,0],data_domain[data_domain[:, 2] == 0.1][:,1])
plt.show()