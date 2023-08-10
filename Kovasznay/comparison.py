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


class Net(nn.Module):
    def __init__(self, hidden_layers, nodes_per_layer, activation_function):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        # 添加输入层到第一个隐藏层的连接
        self.layers.append(nn.Linear(2, nodes_per_layer))
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


device = torch.device("cuda")	# 使用gpu训练
net = torch.load('/home/user/ZHOU-Wen/PINN/demo/Kovasznay/best_net_with_improved(all improvements).pth')
# net = torch.load('/home/user/ZHOU-Wen/PINN/demo/Kovasznay/best_net_with_improved(DE).pth')
# net = torch.load('/home/user/ZHOU-Wen/PINN/demo/Kovasznay/best_net_with_improved(adaptive loss).pth')
# net = torch.load('/home/user/ZHOU-Wen/PINN/demo/Kovasznay/best_net_with_improved(adaptive sampling).pth')
# net = torch.load('/home/user/ZHOU-Wen/PINN/demo/Kovasznay/best_net_no_improved.pth')


mesh_x = 300
mesh_y = 300

x = np.linspace(-0.5,1,mesh_x)
y = np.linspace(-0.5,1.5,mesh_y)
ms_x, ms_y = np.meshgrid(x, y)
x = np.ravel(ms_x).reshape(-1, 1)
y = np.ravel(ms_y).reshape(-1, 1)

pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
pt_y = Variable(torch.from_numpy(y).float(), requires_grad=True).to(device)

predict = net(torch.cat([pt_x, pt_y], 1)).to(device)
pre_u, pre_v, pre_p = predict[:,0].cpu().data.numpy(),predict[:,1].cpu().data.numpy(),predict[:,2].cpu().data.numpy()

pre_u = pre_u.reshape(mesh_y,mesh_x)
pre_u = np.flipud(pre_u)

pre_v = pre_v.reshape(mesh_y,mesh_x)
pre_v = np.flipud(pre_v)

pre_p = pre_p.reshape(mesh_y,mesh_x)
pre_p = np.flipud(pre_p)


Re = 40
lam = 0.5 * Re - np.sqrt(0.25 * (Re ** 2) + 4 * (np.pi ** 2))

x = np.linspace(-0.5, 1, mesh_x)
y = np.linspace(-0.5, 1.5, mesh_y)
real_u = []
real_v = []
real_p = []
for i in range(len(y)):
    for j in range(len(x)):
        real_u.append(1 - np.exp(lam * x[j]) * np.cos(2 * np.pi * y[i]))
        real_v.append(lam / (2 * np.pi) * np.exp(lam * x[j]) * np.sin(2 * np.pi * y[i]))
        real_p.append(1 / 2 * (1 - np.exp(2 * lam * x[j])))


real_u = np.array(real_u).reshape(mesh_y,mesh_x)
real_u = np.flipud(real_u)
real_v = np.array(real_v).reshape(mesh_y,mesh_x)
real_v = np.flipud(real_v)
real_p = np.array(real_p).reshape(mesh_y,mesh_x)
real_p = np.flipud(real_p)

# 相对误差
def mean_relative_error(y_pred, y_true,):
    assert y_pred.shape == y_true.shape, "两个矩阵的形状必须相同"
    assert np.all(y_true != 0), "矩阵B中不能有零元素"

    relative_error = np.abs((y_pred - y_true) / y_true)
    mean_relative_error = np.mean(relative_error)

    return mean_relative_error
print(mean_relative_error(real_u,pre_u))
print(mean_relative_error(real_v,pre_v))
print(mean_relative_error(real_p,pre_p))

def relative_error(y_true, y_pred,):
    relative_error = (y_true - y_pred) / y_true
    return relative_error

comparison = relative_error(real_u,pre_u)

comparison = np.array(comparison).reshape(mesh_y,mesh_x)
# comparison = np.flipud(comparison)
# Create a dataset
comparison = pd.DataFrame(comparison)
# print(comparison)

# Default heatmap
fig = plt.figure()
plt.title("MAPE",size=15)
ax = sns.heatmap(data=comparison,cmap="rainbow",robust=True,vmin=-0.1,vmax=0.1,xticklabels=False,yticklabels=False)
ax.set_xticks(range(0, 300, 20))
ax.set_xticklabels(f'{c:.1f}' for c in np.arange(-0.5, 1.0, 0.1))
ax.set_yticks(range(0, 300, 15))
ax.set_yticklabels(f'{c:.1f}' for c in np.arange(1.5, -0.5, -0.1))

plt.show()


