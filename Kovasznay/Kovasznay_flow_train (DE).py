import pandas as pd
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import time
import csv
import os
import random
from torch.utils.data import DataLoader, TensorDataset
import torch_optimizer as optim
from scipy.stats import qmc



def seed_everything(seed):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(1)


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


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def NS_Kovasznay(x, y, net):
    predict = net(torch.cat([x, y], 1))
    # reshape is very important and necessary
    u, v, p = predict[:, 0].reshape(-1, 1), predict[:, 1].reshape(-1, 1), predict[:, 2].reshape(-1, 1)

    u_x = gradients(u, x)
    u_y = gradients(u, y)
    u_xx = gradients(u, x, 2)
    u_yy = gradients(u, y, 2)

    v_x = gradients(v, x)
    v_y = gradients(v, y)
    v_xx = gradients(v, x, 2)
    v_yy = gradients(v, y, 2)

    p_x = gradients(p, x)
    p_y = gradients(p, y)

    f_u = ((u * u_x + v * u_y) + p_x - (1.0 / 40) * (u_xx + u_yy))
    f_v = ((u * v_x + v * v_y) + p_y - (1.0 / 40) * (v_xx + v_yy))

    f_e = u_x + v_y

    return f_u, f_v, f_e


def gradients(y, x, order=1):
    if order == 1:
        return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True, )[0]
    else:
        return gradients(gradients(y, x), x, order=order - 1)


device = torch.device("cuda")  # 使用gpu训练

# if (os.path.exists('loss_with.csv')):
#     # 存在，则删除文件
#     os.remove('loss_with.csv')
# header = ['loss','mse_f1', 'mse_f2', 'mse_f3', 'mse_u_bc', 'mse_v_bc', 'mse_p_bc','a','b','c','d','e','f']
# with open('loss_with.csv', 'a', encoding='utf-8', newline='') as fp:
#     writer = csv.writer(fp)
#     writer.writerow(header)

net = Net(6, 60, 2)
net.apply(weights_init).to(device)
Re = 40
lam = 0.5 * Re - np.sqrt(0.25 * (Re ** 2) + 4 * (np.pi ** 2))

# BC
# x_bc指的是流域的四条边，其他同理
x_interval = 10
y_interval = 10
x_bc = np.linspace(-0.5, 1.0, x_interval + 1)
y_bc = np.linspace(-0.5, 1.5, y_interval + 1)
x_bc = np.concatenate([np.array([-0.5] * x_interval), np.array([1] * x_interval), x_bc[0:x_interval], x_bc[1:x_interval + 1]], 0)
y_bc = np.concatenate([y_bc[1:y_interval + 1], y_bc[0:y_interval], np.array([-0.5] * y_interval), np.array([1.5] * y_interval)], 0)
x_bc = x_bc.reshape(x_bc.shape[0], 1)
y_bc = y_bc.reshape(y_bc.shape[0], 1)

u_bc = 1 - np.exp(lam * x_bc) * np.cos(2 * np.pi * y_bc)
v_bc = lam / (2 * np.pi) * np.exp(lam * x_bc) * np.sin(2 * np.pi * y_bc)
p_bc = 1 / 2 * (1 - np.exp(2 * lam * x_bc))

x_bc = Variable(torch.from_numpy(x_bc).float(), requires_grad=False).to(device)
y_bc = Variable(torch.from_numpy(y_bc).float(), requires_grad=False).to(device)

u_bc = Variable(torch.from_numpy(u_bc).float(), requires_grad=False).to(device)
v_bc = Variable(torch.from_numpy(v_bc).float(), requires_grad=False).to(device)
p_bc = Variable(torch.from_numpy(p_bc).float(), requires_grad=False).to(device)

# PDE
x_pde = np.random.uniform(low=-0.5, high=1.0, size=(100, 1))
y_pde = np.random.uniform(low=-0.5, high=1.5, size=(100, 1))

# convert pytorch Variable
pt_x_collocation = Variable(torch.from_numpy(x_pde).float(), requires_grad=True).to(device)
pt_y_collocation = Variable(torch.from_numpy(y_pde).float(), requires_grad=True).to(device)

mse_cost_function = torch.nn.MSELoss(reduction='mean')  # Mean squared error
optimizer = torch.optim.Adam(net.parameters())
# optimizer = optim.Adahessian(params, lr= 1.0,betas= (0.9, 0.999),eps= 1e-4,weight_decay=0.0,hessian_power=1.0,)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=200, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

iterations = 10000
min_loss = 10000000
time1 = time.time()
print(time1)

for epoch in range(iterations):

    # Loss based on boundary conditions
    predict = net(torch.cat([x_bc, y_bc], 1))  # output of u(t,x)
    u, v, p = predict[:, 0].reshape(-1, 1), predict[:, 1].reshape(-1, 1), predict[:, 2].reshape(-1, 1)
    mse_u_bc = mse_cost_function(u, u_bc)
    mse_v_bc = mse_cost_function(v, v_bc)
    mse_p_bc = mse_cost_function(p, p_bc)
    #
    # # Loss based on initial value
    # net_ini_out1 = net(torch.cat([ pt_t_ini,pt_x_ini],1)) # output of u(t,x)
    # net_ini_out2 = u_t_grad(torch.cat([ pt_t_ini,pt_x_ini],1))
    # mse_u3 = mse_cost_function(net_ini_out1, pt_u_ini)
    # mse_u4 = mse_cost_function(net_ini_out2, pt_u_ini)

    # Loss based on PDE
    f_u, f_v, f_e = NS_Kovasznay(pt_x_collocation, pt_y_collocation, net=net)  # output of f(t,x)
    mse_f1 = mse_cost_function(f_u, torch.zeros_like(f_u))
    mse_f2 = mse_cost_function(f_v, torch.zeros_like(f_v))
    mse_f3 = mse_cost_function(f_e, torch.zeros_like(f_e))

    # Combining the loss functions
    loss = mse_f1 + mse_f2 + mse_f3 + mse_u_bc + mse_v_bc + mse_p_bc
    optimizer.zero_grad()  # to make the gradients zero
    loss.backward()  # This is for computing gradients using backward propagation
    optimizer.step()  # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta
    scheduler.step(loss)

    if loss < min_loss:
        min_loss = loss
        # 保存模型语句
        torch.save(net, '/home/user/ZHOU-Wen/PINN/demo/Kovasznay/best_net_with_improved(DE).pth')
        print('current epoch', epoch)
    # with torch.autograd.no_grad():
    #     if epoch % 1 == 0:
    #         data = [loss.item(),
    #                 mse_cost_function(f_u, torch.zeros_like(f_u)).item(),
    #                 mse_cost_function(f_v, torch.zeros_like(f_v)).item(),
    #                 mse_cost_function(f_e, torch.zeros_like(f_e)).item(),
    #                 mse_cost_function(u, u_bc).item(), mse_cost_function(v, v_bc).item(),
    #                 mse_cost_function(p, p_bc).item(), log_var_a.item(), log_var_b.item(), log_var_c.item(),
    #                 log_var_d.item(), log_var_e.item(), log_var_f.item()
    #                 ]
    #         with open('loss_with.csv', 'a', encoding='utf-8', newline='') as fp:
    #             # 写
    #             writer = csv.writer(fp)
    #             # 将数据写入
    #             writer.writerow(list(data))
    if epoch % 500 == 0:
        print(epoch, "Traning Loss:", loss.item())

print(min_loss)
time2 = time.time()
print(time2)
print('userd: {:.5f}s'.format(time2 - time1))

