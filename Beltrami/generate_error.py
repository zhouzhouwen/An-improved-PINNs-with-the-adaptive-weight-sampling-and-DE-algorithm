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
        self.input_layer = nn.Linear(2, NN)
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

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)



# class Net(nn.Module):
#     def __init__(self, input_size=4, output_size=4):
#         super(Net, self).__init__()
#         self.layer = nn.Sequential(nn.Linear(input_size, 150),
#                                    # 字符型的变量后面不跟0，跟上一行区别
#                                    # eval(act1),
#                                    nn.Tanh(),
#                                    nn.Linear(150, 60),
#                                    # eval(act2),
#                                    nn.Tanh(),
#                                    nn.Linear(60, 100),
#                                    # eval(act3),
#                                    nn.Tanh(),
#                                    nn.Linear(100, 80),
#                                    # eval(act4),
#                                    nn.Tanh(),
#                                    nn.Linear(80, 150),
#                                    # eval(act5),
#                                    nn.Tanh(),
#                                    nn.Linear(150, 130),
#                                    # eval(act6),
#                                    nn.Tanh(),
#                                    nn.Linear(130, 160),
#                                    nn.Tanh(),
#                                    nn.Linear(160, 90),
#                                    nn.Tanh(),
#                                    nn.Linear(90, output_size),
#                                    )
#
#     def forward(self, x):
#         x = self.layer(x)
#         return x

# class Net(nn.Module):
#     def __init__(self, hidden_layers, nodes_per_layer, activation_function):
#         super(Net, self).__init__()
#         self.layers = nn.ModuleList()
#         # 添加输入层到第一个隐藏层的连接
#         self.layers.append(nn.Linear(4, nodes_per_layer))
#         # 添加隐藏层
#         for _ in range(hidden_layers):
#             self.layers.append(nn.Linear(nodes_per_layer, nodes_per_layer))
#
#         # 添加最后一层到输出层的连接
#         self.layers.append(nn.Linear(nodes_per_layer, 4))
#
#         # 设置激活函数
#         if activation_function == 0:
#             self.activation = nn.Softsign()
#         elif activation_function == 1:
#             self.activation = nn.Softplus()
#         elif activation_function == 2:
#             self.activation = nn.Tanh()
#         elif activation_function == 3:
#             self.activation = nn.Tanhshrink()
#         elif activation_function == 4:
#             self.activation = nn.ReLU()
#         elif activation_function == 5:
#             self.activation = nn.RReLU()
#         else:
#             raise ValueError("Invalid activation function identifier")
#     def forward(self, x):
#         for layer in self.layers[:-1]:
#             x = self.activation(layer(x))
#         # 最后一层不使用激活函数
#         x = self.layers[-1](x)
#         return x

# net = torch.load('/home/user/ZHOU-Wen/PINN/demo/Beltrami/best_net_with_improved(all improvements).pth')
# net = torch.load('/home/user/ZHOU-Wen/PINN/demo/Beltrami/best_net_with_improved(DE).pth')
# net = torch.load('/home/user/ZHOU-Wen/PINN/demo/Beltrami/best_net_with_improved(adaptive sampling).pth')
# net = torch.load('/home/user/ZHOU-Wen/PINN/demo/Beltrami/best_net_with_improved(adaptive loss).pth')
# net = torch.load('/home/user/ZHOU-Wen/PINN/demo/Beltrami/best_net_no_improved.pth')


device = torch.device("cuda")	# 使用gpu训练
# 相对误差
def mean_relative_error(y_true, y_pred,):
    relative_error = np.average(np.abs(y_true - y_pred) / np.abs(y_true), axis=0)
    return relative_error

def relative_error(y_true, y_pred,):
    relative_error = (y_true - y_pred) / y_true
    return relative_error

a, d = 1, 1
def data_generate(x, y, z, t):
    u = - a * (np.exp(a * x) * np.sin(a * y + d * z) + np.exp(a * z) * np.cos(a * x + d * y)) * np.exp(- d * d * t)
    v = - a * (np.exp(a * y) * np.sin(a * z + d * x) + np.exp(a * x) * np.cos(a * y + d * z)) * np.exp(- d * d * t)
    w = - a * (np.exp(a * z) * np.sin(a * x + d * y) + np.exp(a * y) * np.cos(a * z + d * x)) * np.exp(- d * d * t)
    p = - 0.5 * a * a * (np.exp(2 * a * x) + np.exp(2 * a * y) + np.exp(2 * a * z) +
                         2 * np.sin(a * x + d * y) * np.cos(a * z + d * x) * np.exp(a * (y + z)) +
                         2 * np.sin(a * y + d * z) * np.cos(a * x + d * y) * np.exp(a * (z + x)) +
                         2 * np.sin(a * z + d * x) * np.cos(a * y + d * z) * np.exp(a * (x + y))) * np.exp(
        -2 * d * d * t)

    return u, v, w, p



mesh_x = 31
mesh_y = 31
mesh_z = 31
x = np.linspace(-1, 1, mesh_x)
y = np.linspace(-1, 1, mesh_y)
z = np.linspace(-1, 1, mesh_z)
ms_x, ms_y, ms_z = np.meshgrid(x, y, z)
x = np.ravel(ms_x).reshape(-1, 1)
y = np.ravel(ms_y).reshape(-1, 1)
z = np.ravel(ms_z).reshape(-1, 1)

error_u = []
error_v = []
error_w = []
error_p = []
for i in range(0,11):
    t = i/10 * np.ones((mesh_x*mesh_y*mesh_z, 1))

    real_u, real_v, real_w, real_p = data_generate(x,y,z,t)

    pt_x = Variable(torch.from_numpy(x).float(), requires_grad=False).to(device)
    pt_y = Variable(torch.from_numpy(y).float(), requires_grad=False).to(device)
    pt_z = Variable(torch.from_numpy(z).float(), requires_grad=False).to(device)
    pt_t = Variable(torch.from_numpy(t).float(), requires_grad=False).to(device)
    predict = net(torch.cat([pt_x, pt_y, pt_z, pt_t], 1)).to(device)
    pre_u, pre_v, pre_w, pre_p = predict[:,0].cpu().data.numpy().reshape(-1,1),predict[:,1].cpu().data.numpy().reshape(-1,1),predict[:,2].cpu().data.numpy().reshape(-1,1),predict[:,3].cpu().data.numpy().reshape(-1,1)

    error_u.append(mean_relative_error(pre_u, real_u))
    error_v.append(mean_relative_error(pre_v, real_v))
    error_w.append(mean_relative_error(pre_w, real_w))
    error_p.append(mean_relative_error(pre_p, real_p))

error_no_improved = np.concatenate([np.array(error_u),np.array(error_v),np.array(error_w),np.array(error_p)],1)
print(np.average(error_u))
print(np.average(error_v))
print(np.average(error_w))
print(np.average(error_p))
# print(error_v)
# print(error_w)
# print(error_p)
# print(error_no_improved)
np.save('error_no_improved.npy',error_no_improved)





