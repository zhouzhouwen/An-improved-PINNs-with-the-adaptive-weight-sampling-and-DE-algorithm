import pandas as pd
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import time

class Net(nn.Module):
    def __init__(self, NL, NN):
        # NL是有多少层隐藏层
        # NN是每层的神经元数量
        super(Net, self).__init__()
        self.input_layer = nn.Linear(3, NN)
        self.hidden_layer = nn.ModuleList([nn.Linear(NN, NN) for i in range(NL)])
        self.output_layer = nn.Linear(NN, 3)

    def forward(self, x):
        o = self.act(self.input_layer(x))
        for i, li in enumerate(self.hidden_layer):
            o = self.act(li(o))
        out = self.output_layer(o)
        return out

    def act(self, x):
        return torch.tanh(x)

#
# class Net(nn.Module):
#     def __init__(self, input_size=3, output_size=3):
#         super(Net, self).__init__()
#         self.layer = nn.Sequential()
#
#     def forward(self, x):
#         x = self.layer(x)
#         return x

class Net(nn.Module):
    def __init__(self, hidden_layers, nodes_per_layer, activation_function):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        # 添加输入层到第一个隐藏层的连接
        self.layers.append(nn.Linear(3, nodes_per_layer))
        # 添加隐藏层
        for _ in range(hidden_layers):
            self.layers.append(nn.Linear(nodes_per_layer, nodes_per_layer))

        # 添加最后一层到输出层的连接
        self.layers.append(nn.Linear(nodes_per_layer, 3))

        # 设置激活函数
        if activation_function == 0:
            self.activation = nn.Softsign()
        elif activation_function == 1:
            self.activation = nn.Softplus()
        elif activation_function == 2:
            self.activation = nn.Tanh()
        elif activation_function == 3:
            self.activation = nn.Tanhshrink()
        elif activation_function == 4:
            self.activation = nn.ReLU()
        elif activation_function == 5:
            self.activation = nn.RReLU()
        else:
            raise ValueError("Invalid activation function identifier")

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        # 最后一层不使用激活函数
        x = self.layers[-1](x)
        return x

# net = torch.load('/home/user/ZHOU-Wen/PINN/demo/cylinder_wake/best_net_no_improved.pth')
# net = torch.load('/home/user/ZHOU-Wen/PINN/demo/cylinder_wake/best_net_with_improved(adaptive loss).pth')
# net = torch.load('/home/user/ZHOU-Wen/PINN/demo/cylinder_wake/best_net_with_improved(adaptive resampling).pth')
# net = torch.load('/home/user/ZHOU-Wen/PINN/demo/cylinder_wake/best_net_with_improved(DE).pth')
net = torch.load('/home/user/ZHOU-Wen/PINN/demo/cylinder_wake/best_net_with_improved(all improvements)1.pth')
#

device = torch.device("cuda")	# 使用gpu训练
# 相对误差
def mean_relative_error(y_pred, y_true,):
    assert y_pred.shape == y_true.shape, "两个矩阵的形状必须相同"
    assert np.all(y_true != 0), "矩阵B中不能有零元素"

    relative_error = np.abs((y_pred - y_true) / y_true)
    mean_relative_error = np.mean(relative_error)
    return mean_relative_error

# def mean_relative_error(y_true, y_pred,):
#     relative_error = np.average(np.abs(y_true - y_pred) / np.abs(y_true), axis=0)
#     return relative_error

def relative_error(y_true, y_pred,):
    relative_error = (y_true - y_pred) / y_true
    return relative_error


data_domain = np.load('data_domain.npy')

data_domain = data_domain[data_domain[:, 2].argsort()] # 按第2列进行排
# print(data_domain)
x = data_domain[:,0].reshape(-1,1)
y = data_domain[:,1].reshape(-1,1)
t = data_domain[:,2].reshape(-1,1)
pt_x = Variable(torch.from_numpy(x).float(), requires_grad=False).to(device)
pt_y = Variable(torch.from_numpy(y).float(), requires_grad=False).to(device)
pt_t = Variable(torch.from_numpy(t).float(), requires_grad=False).to(device)

real_u = data_domain[:, 3].ravel()
real_v = data_domain[:, 4].ravel()
real_p = data_domain[:, 5].ravel()

error_u = []
error_v = []
error_p = []

predict = net(torch.cat([pt_x, pt_y, pt_t], 1)).to(device)
pre_u, pre_v, pre_p = predict[:,0].cpu().data.numpy().reshape(-1,1), predict[:,1].cpu().data.numpy().reshape(-1,1), predict[:,2].cpu().data.numpy().reshape(-1,1)
pre_u = pre_u.ravel()
pre_v = pre_v.ravel()
pre_p = pre_p.ravel()

# print(pre_u[55000])
for i in range(0,20):
    error_u.append(mean_relative_error(pre_u[5000*i:5000*(i+1)], real_u[5000*i:5000*(i+1)]))
    error_v.append(mean_relative_error(pre_v[5000*i:5000*(i+1)], real_v[5000*i:5000*(i+1)]))
    error_p.append(mean_relative_error(pre_p[5000*i:5000*(i+1)], real_p[5000*i:5000*(i+1)]))


# print(error_u)
# print(error_v)
# print(error_p)
error_no_improved = np.concatenate([np.array(error_u).reshape(-1,1),np.array(error_v).reshape(-1,1),np.array(error_p).reshape(-1,1)],1)
# print(error_no_improved)
# 计算每列的平均值
column_means = error_no_improved.mean(axis=0)
print(column_means)
np.save('error_with_improved.npy',error_no_improved)





