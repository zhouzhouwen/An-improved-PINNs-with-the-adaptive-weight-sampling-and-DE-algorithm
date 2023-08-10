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
#         self.layer = nn.Sequential(nn.Linear(input_size, 130),
#                                    # 字符型的变量后面不跟0，跟上一行区别
#                                    # eval(act1),
#                                    nn.Tanh(),
#                                    nn.Linear(130, 130),
#                                    # eval(act2),
#                                    nn.Tanh(),
#                                    nn.Linear(130, 80),
#                                    # eval(act3),
#                                    nn.Tanh(),
#                                    nn.Linear(80, 120),
#                                    # eval(act4),
#                                    nn.Tanh(),
#                                    nn.Linear(120, 130),
#                                    # eval(act5),
#                                    nn.Tanh(),
#                                    nn.Linear(130, 100),
#                                    # eval(act6),
#                                    nn.Tanh(),
#                                    nn.Linear(100, 120),
#                                    nn.Tanh(),
#                                    nn.Linear(120, 60),
#                                    nn.Tanh(),
#                                    nn.Linear(60, output_size),
#                                    )
#
#     def forward(self, x):
#         x = self.layer(x)
#         return x
#
# def weights_init(m):
#     if isinstance(m, nn.Linear):
#         nn.init.xavier_uniform_(m.weight)
#         nn.init.zeros_(m.bias)
#     if isinstance(m, nn.Conv2d):
#         nn.init.xavier_uniform_(m.weight)
#         nn.init.zeros_(m.bias)

class Net(nn.Module):
    def __init__(self, hidden_layers, nodes_per_layer, activation_function):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        # 添加输入层到第一个隐藏层的连接
        self.layers.append(nn.Linear(4, nodes_per_layer))
        # 添加隐藏层
        for _ in range(hidden_layers):
            self.layers.append(nn.Linear(nodes_per_layer, nodes_per_layer))

        # 添加最后一层到输出层的连接
        self.layers.append(nn.Linear(nodes_per_layer, 4))

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

# net = torch.load('/home/user/ZHOU-Wen/PINN/demo/Beltrami/best_net_no_improved.pth')
net = torch.load('/home/user/ZHOU-Wen/PINN/demo/Beltrami/best_net_with_improved(all improvements).pth')

mesh_x = 31
mesh_y = 31
x = np.linspace(-1,1,mesh_x)
y = np.linspace(-1,1,mesh_y)
ms_x, ms_y = np.meshgrid(x, y)
x = np.ravel(ms_x).reshape(-1, 1)
y = np.ravel(ms_y).reshape(-1, 1)
z = 0.5 * np.ones((mesh_x*mesh_y, 1))
t = 1.0 * np.ones((mesh_x*mesh_y, 1))

pt_x = Variable(torch.from_numpy(x).float(), requires_grad=False).to(device)
pt_y = Variable(torch.from_numpy(y).float(), requires_grad=False).to(device)
pt_z = Variable(torch.from_numpy(z).float(), requires_grad=False).to(device)
pt_t = Variable(torch.from_numpy(t).float(), requires_grad=False).to(device)

predict = net(torch.cat([pt_x, pt_y, pt_z, pt_t], 1)).to(device)
u, v, w, p = predict[:,0].cpu().data.numpy(),predict[:,1].cpu().data.numpy(),predict[:,2].cpu().data.numpy(),predict[:,3].cpu().data.numpy()

pt_u=u.reshape(mesh_x,mesh_y)
pt_u = np.flipud(pt_u)
# pt_u = np.rot90(pt_u, -3)

pt_v=v.reshape(mesh_x,mesh_y)
pt_v = np.flipud(pt_v)

pt_w=w.reshape(mesh_x,mesh_y)
pt_w = np.flipud(pt_w)

pt_p=p.reshape(mesh_x,mesh_y)
pt_p = np.rot90(pt_p, -3)

# Create a dataset
df_u = pd.DataFrame(pt_u)
df_v = pd.DataFrame(pt_v)
df_w = pd.DataFrame(pt_w)
df_p = pd.DataFrame(pt_p)

# Default heatmap
fig = plt.figure(figsize=(12,9))
plt.title("Improved PINN u(x,y,z)",size=25)
ax = sns.heatmap(data=df_u, cmap="rainbow",robust=True,xticklabels=False,yticklabels=False)
ax.set_xticks(range(0, 31, 2))
ax.set_xticklabels(f'{c:.2f}' for c in np.arange(-1.0, 1.0, 2/16))
ax.set_yticks(range(0, 31, 2))
ax.set_yticklabels(f'{c:.2f}' for c in np.arange(1.0, -1.0, -2/16))
plt.yticks(size=10,)
plt.xticks(size=10,)
plt.show()