import pandas as pd
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import time

class Net(nn.Module):
    def __init__(self, input_size=3, output_size=3):
        super(Net, self).__init__()
        self.layer = nn.Sequential()

    def forward(self, x):
        x = self.layer(x)
        return x


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

device = torch.device("cuda")	# 使用gpu训练

net=torch.load('/home/user/ZHOU-Wen/PINN/demo/cylinder_wake/best_net_with_improved3.pth.pth')

real_data = np.load('data_domain.npy')
real_data = real_data[real_data[:,2] == 1.0]

x = real_data[:, 0].reshape(-1,1)
y = real_data[:, 1].reshape(-1,1)
t = real_data[:, 2].reshape(-1,1)
real_u = real_data[:, 3]
real_v = real_data[:, 4]
real_p = real_data[:, 5]

pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
pt_y = Variable(torch.from_numpy(y).float(), requires_grad=True).to(device)
pt_t = Variable(torch.from_numpy(t).float(), requires_grad=True).to(device)

predict = net(torch.cat([pt_x, pt_y, pt_t], 1)).to(device)
pre_u, pre_v, pre_p = predict[:, 0].cpu().data.numpy(), predict[:, 1].cpu().data.numpy(), predict[:, 2].cpu().data.numpy()

pre_u = pre_u.reshape(50,100)
pre_u = np.flipud(pre_u)
pre_v = pre_v.reshape(50,100)
pre_v = np.flipud(pre_v)
pre_p = pre_p.reshape(50,100)
pre_p = np.flipud(pre_p)

# Create a dataset
pre_u = pd.DataFrame(pre_u)
pre_v = pd.DataFrame(pre_v)
pre_p = pd.DataFrame(pre_p)

# Default heatmap
fig = plt.figure(figsize=(7, 4))
ax = sns.heatmap(data=pre_u,cmap="rainbow",xticklabels=False,yticklabels=False)

plt.show()

def mean_relative_error(y_true, y_pred):
    relative_error = np.average(np.abs((y_true - y_pred) / y_true), axis=0)
    return relative_error




# 1000的意思是，有1000个值，67是要被分为多少段，67*15 等于1000差不多，y就是50*20咯
# ax.set_xticks(range(0, 1000, 67))
# ax.set_xticklabels(f'{c:.1f}' for c in np.arange(-0.5, 1.0, 0.1))
# ax.set_yticks(range(0, 1000, 50))
# ax.set_yticklabels(f'{c:.1f}' for c in np.arange(1.5, -0.5, -0.1))

# plt.title('Results', fontsize=20)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.xlabel('x', fontsize=18)
# plt.ylabel('y', fontsize=18)
#ax_v = sns.heatmap(data=df_v,cmap="rainbow",center=0,robust=True,xticklabels=False,yticklabels=False)
#ax_p = sns.heatmap(data=df_p,cmap="rainbow",center=0,robust=True,xticklabels=False,yticklabels=False)
