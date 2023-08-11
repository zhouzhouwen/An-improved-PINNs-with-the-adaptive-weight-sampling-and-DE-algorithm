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
from torch.cuda.amp import autocast, GradScaler


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

seed_everything(3)

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

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def NS_cylinder(x, y, t, net):

    predict = net(torch.cat([x, y, t], 1))
    # reshape is very important and necessary
    u, v, p = predict[:,0].reshape(-1, 1), predict[:,1].reshape(-1, 1), predict[:,2].reshape(-1, 1)

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

    u_t = gradients(u, t)
    v_t = gradients(v, t)

    f_u = (u_t + (u * u_x + v * u_y) + p_x - 0.01 * (u_xx + u_yy))
    f_v = (v_t + (u * v_x + v * v_y) + p_y - 0.01 * (v_xx + v_yy))

    f_e = u_x + v_y

    return f_u, f_v, f_e


def gradients(y, x, order=1):
    if order == 1:
        return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True, )[0]
    else:
        return gradients(gradients(y, x), x, order=order - 1)

device = torch.device("cuda")  # 使用gpu训练


if (os.path.exists('loss_with.csv')):
    # 存在，则删除文件
    os.remove('loss_with.csv')
header = ['loss','mse_f1','mse_f2','mse_f3', 'mse_u_bc' ,'mse_v_bc' ,'mse_p_bc' ,'mse_u_ic' ,'mse_v_ic' ,'mse_p_ic',
          'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',]
with open('loss_with.csv', 'a', encoding='utf-8', newline='') as fp:
    writer = csv.writer(fp)
    writer.writerow(header)

net = Net(10, 100, 2)
net.apply(weights_init).to(device)
net.apply(weights_init).to(device)

# BC
# x_bc指的是流域的四条边，其他同理
# PS: y is the edge of x
data_bc = np.load('data_bc.npy')
x_bc = Variable(torch.from_numpy(data_bc[:, 0].reshape(-1, 1)).float(), requires_grad=False).to(device)
y_bc = Variable(torch.from_numpy(data_bc[:, 1].reshape(-1, 1)).float(), requires_grad=False).to(device)
t_bc = Variable(torch.from_numpy(data_bc[:, 2].reshape(-1, 1)).float(), requires_grad=False).to(device)
u_bc = Variable(torch.from_numpy(data_bc[:, 3].reshape(-1, 1)).float(), requires_grad=False).to(device)
v_bc = Variable(torch.from_numpy(data_bc[:, 4].reshape(-1, 1)).float(), requires_grad=False).to(device)
p_bc = Variable(torch.from_numpy(data_bc[:, 5].reshape(-1, 1)).float(), requires_grad=False).to(device)

# IC
data_ic = np.load('data_ic.npy')
x_ic = Variable(torch.from_numpy(data_ic[:, 0].reshape(-1, 1)).float(), requires_grad=False).to(device)
y_ic = Variable(torch.from_numpy(data_ic[:, 1].reshape(-1, 1)).float(), requires_grad=False).to(device)
t_ic = Variable(torch.from_numpy(data_ic[:, 2].reshape(-1, 1)).float(), requires_grad=False).to(device)
u_ic = Variable(torch.from_numpy(data_ic[:, 3].reshape(-1, 1)).float(), requires_grad=False).to(device)
v_ic = Variable(torch.from_numpy(data_ic[:, 4].reshape(-1, 1)).float(), requires_grad=False).to(device)
p_ic = Variable(torch.from_numpy(data_ic[:, 5].reshape(-1, 1)).float(), requires_grad=False).to(device)

# PDE
x_pde = np.load('resample_point_x.npy')
y_pde = np.load('resample_point_y.npy')
t_pde = np.load('resample_point_t.npy')

# 根据索引对三个数组进行打乱
# indices = np.random.permutation(x_pde.shape[0])
# x_pde = x_pde[indices]
# y_pde = y_pde[indices]
# t_pde = t_pde[indices]

# data_domain = np.load('data_domain.npy')
# print(data_domain.shape)
# idx = np.load('first_sample_point_sobol.npy')
# x_pde = data_domain[idx, 0].reshape(-1, 1)
# y_pde = data_domain[idx, 1].reshape(-1, 1)
# t_pde = data_domain[idx, 2].reshape(-1, 1)


# convert pytorch Variable
x_pde = Variable(torch.from_numpy(x_pde).float(), requires_grad=True).to(device)
y_pde = Variable(torch.from_numpy(y_pde).float(), requires_grad=True).to(device)
t_pde = Variable(torch.from_numpy(t_pde).float(), requires_grad=True).to(device)

log_var_a = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
log_var_b = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
log_var_c = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
log_var_d = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
log_var_e = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
log_var_f = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
log_var_g = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
log_var_h = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
log_var_i = Variable(1 * torch.ones(1).cuda(), requires_grad=True)

# get all parameters (model parameters + task dependent log variances)
params = ([p for p in net.parameters()] + [log_var_a] + [log_var_b] + [log_var_c] + [log_var_d] + [log_var_e] + [log_var_f] + [log_var_g] + [log_var_h] + [log_var_i])

mse_cost_function = torch.nn.MSELoss(reduction='mean')  # Mean squared error
optimizer = torch.optim.Adam(params, lr=0.001)
# optimizer = optim.Adahessian(params, lr= 1.0,betas= (0.9, 0.999),eps= 1e-4,weight_decay=0.0,hessian_power=1.0,)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20000, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

# define loss criterion
def criterion(y_pred, y_true, log_vars):
    loss = 0
    # method 1
    # for i in range(len(y_pred)):
    #     precision = torch.exp(-log_vars)
    #     diff = (y_pred - y_true) ** 2.0
    #     loss += torch.sum(precision * diff + log_vars, -1)
    # method 2
    # for i in range(len(y_pred)):
    #     precision = 0.5 / (log_vars ** 2)
    #     diff = (y_pred - y_true) ** 2.0
    #     loss += torch.sum(precision * diff + torch.log(1 + log_vars ** 2), -1)

    precision = 0.5 / (log_vars ** 2)
    diff = (y_pred - y_true) ** 2.0
    loss += torch.sum(precision * diff + torch.log(1 + log_vars ** 2), -1)

    return torch.mean(loss)

iterations = 3 * 10 ** 5
min_loss_pde = 10000000
min_loss_bc = 10000000
min_loss_ic = 10000000
min_loss = 10000000
time1 = time.time()
print(time1)
scaler = GradScaler()
for epoch in range(iterations):
    optimizer.zero_grad()  # to make the gradients zero
    with autocast():
        # Loss based on boundary conditions
        predict_bc = net(torch.cat([x_bc, y_bc, t_bc], 1))
        predict_u_bc, predict_v_bc, predict_p_bc = predict_bc[:, 0].reshape(-1, 1), predict_bc[:, 1].reshape(-1, 1), predict_bc[:, 2].reshape(-1, 1)
        mse_u_bc = criterion(predict_u_bc, u_bc, log_var_a)
        mse_v_bc = criterion(predict_v_bc, v_bc, log_var_b)
        mse_p_bc = criterion(predict_p_bc, p_bc, log_var_c)
        #
        # # Loss based on initial value
        predict_ic = net(torch.cat([x_ic, y_ic, t_ic], 1))
        predict_u_ic, predict_v_ic, predict_p_ic = predict_ic[:, 0].reshape(-1, 1), predict_ic[:, 1].reshape(-1, 1), predict_ic[:, 2].reshape(-1, 1)
        mse_u_ic = criterion(predict_u_ic, u_ic, log_var_d)
        mse_v_ic = criterion(predict_v_ic, v_ic, log_var_e)
        mse_p_ic = criterion(predict_p_ic, p_ic, log_var_f)

        # Loss based on PDE
        f_u, f_v, f_e = NS_cylinder(x_pde, y_pde, t_pde, net=net)  # output of f(t,x)
        mse_f1 = criterion(f_u, torch.zeros_like(f_u), log_var_g)
        mse_f2 = criterion(f_v, torch.zeros_like(f_v), log_var_h)
        mse_f3 = criterion(f_e, torch.zeros_like(f_e), log_var_i)

        # Combining the loss functions
        loss = mse_f1 + mse_f2 + mse_f3 + mse_u_bc + mse_v_bc + mse_p_bc + mse_u_ic + mse_v_ic + mse_p_ic

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    # loss.backward()  # This is for computing gradients using backward propagation
    # optimizer.step()  # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta

    # scheduler.step(loss)
    torch.save(net, '/home/user/ZHOU-Wen/PINN/demo/cylinder_wake/last_net_with_improved(all improvements).pth')
    if loss < min_loss:
        min_loss = loss
        # 保存模型语句
        torch.save(net, '/home/user/ZHOU-Wen/PINN/demo/cylinder_wake/best_net_with_improved(all improvements).pth')
        print('current epoch:', epoch)
    # if mse_f1 + mse_f2 + mse_f3 < min_loss_pde and mse_u_bc + mse_v_bc + mse_p_bc<min_loss_bc and mse_u_ic + mse_v_ic + mse_p_ic<min_loss_ic:
    # if loss < min_loss:
    #     min_loss = loss
    #     min_loss_pde = mse_f1 + mse_f2 + mse_f3
    #     min_loss_bc = mse_u_bc + mse_v_bc + mse_p_bc
    #     min_loss_ic = mse_u_ic + mse_v_ic + mse_p_ic
    #     # 保存模型语句
    #     torch.save(net, '/home/user/ZHOU-Wen/PINN/demo/cylinder_wake/best_net_with_improved.pth')
    #     print('current epoch',epoch)
    with torch.autograd.no_grad():
        if epoch % 1 == 0:
            data = [loss.item(),
                    mse_cost_function(f_u, torch.zeros_like(f_u)).item(), mse_cost_function(f_v, torch.zeros_like(f_v)).item(),mse_cost_function(f_e, torch.zeros_like(f_e)).item(),
                    mse_cost_function(predict_u_bc, u_bc).item(), mse_cost_function(predict_v_bc, v_bc).item(),mse_cost_function(predict_p_bc, p_bc).item(),
                    mse_cost_function(predict_u_ic, u_ic).item(), mse_cost_function(predict_v_ic, v_ic).item(), mse_cost_function(predict_p_ic, p_ic).item(),
                    log_var_a.item(), log_var_b.item(), log_var_c.item(), log_var_d.item(), log_var_e.item(), log_var_f.item(), log_var_g.item(), log_var_h.item(), log_var_i.item(),
                    ]
        with open('loss_with.csv', 'a', encoding='utf-8', newline='') as fp:
            # 写
            writer = csv.writer(fp)
            # 将数据写入
            writer.writerow(list(data))
    if epoch % 500 == 0:
        print(epoch, "Traning Loss:", loss.item())

time2 = time.time()
print(time2)
print('userd: {:.5f}s'.format(time2 - time1))
