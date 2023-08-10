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

# net=torch.load('/home/user/ZHOU-Wen/PINN/demo/Kovasznay/best_net_no_improved.pth')
net=torch.load('/home/user/ZHOU-Wen/PINN/demo/Kovasznay/best_net_with_improved(DE).pth')

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
u, v, p = predict[:,0].cpu().data.numpy(),predict[:,1].cpu().data.numpy(),predict[:,2].cpu().data.numpy()

data = p.reshape(mesh_y,mesh_x)
# pt_u = np.rot90(pt_u, -3)
data = np.flipud(data)
data = pd.DataFrame(data)


# Default heatmap
fig = plt.figure()
# ax = sns.heatmap(data=df_u,cmap="rainbow",robust=True)
# ax = sns.heatmap(data=df_v,cmap="rainbow",robust=True,)
ax = sns.heatmap(data=data,cmap="rainbow",robust=True,xticklabels=False,yticklabels=False)

ax.set_xticks(range(0, 300, 20))
ax.set_xticklabels(f'{c:.1f}' for c in np.arange(-0.5, 1.0, 0.1))
ax.set_yticks(range(0, 300, 15))
ax.set_yticklabels(f'{c:.1f}' for c in np.arange(1.5, -0.5, -0.1))

# 1000的意思是，有1000个值，67是要被分为多少段，67*15 等于1000差不多，y就是50*20咯
# ax.set_xticks(range(0, 1000, 67))
# ax.set_xticklabels(f'{c:.1f}' for c in np.arange(-0.5, 1.0, 0.1))
# ax.set_yticks(range(0, 1000, 50))
# ax.set_yticklabels(f'{c:.1f}' for c in np.arange(1.5, -0.5, -0.1))
#
# plt.title('Conventional PINN u(x,y)', fontsize=15)
plt.title('Improved PINN p(x,y)', fontsize=15)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.xlabel('x', fontsize=18)
# plt.ylabel('y', fontsize=18)
#ax_v = sns.heatmap(data=df_v,cmap="rainbow",center=0,robust=True,xticklabels=False,yticklabels=False)
#ax_p = sns.heatmap(data=df_p,cmap="rainbow",center=0,robust=True,xticklabels=False,yticklabels=False)

plt.show()