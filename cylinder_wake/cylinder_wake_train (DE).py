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


# if (os.path.exists('loss_with.csv')):
#     # 存在，则删除文件
#     os.remove('loss_with.csv')
# header = ['loss','mse_f1','mse_f2','mse_f3', 'mse_u_bc' ,'mse_v_bc' ,'mse_p_bc' ,'mse_u_ic' ,'mse_v_ic' ,'mse_p_ic',
#           'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',]
# with open('loss_with.csv', 'a', encoding='utf-8', newline='') as fp:
#     writer = csv.writer(fp)
#     writer.writerow(header)

net = Net(10, 100, 2)
net.apply(weights_init).to(device)
net.apply(weights_init).to(device)
mse_cost_function = torch.nn.MSELoss(reduction='mean') # Mean squared error
# optimizer = torch.optim.LBFGS(net.parameters())
optimizer = torch.optim.Adam(net.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20000, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)


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
data_domain = np.load('data_domain.npy')
print(data_domain.shape[0])
idx = np.random.choice(data_domain.shape[0], 20000, replace=False)

x_pde = data_domain[idx, 0].reshape(-1, 1)
y_pde = data_domain[idx, 1].reshape(-1, 1)
t_pde = data_domain[idx, 2].reshape(-1, 1)

# x_pde = np.load('resample_point_x.npy')
# y_pde = np.load('resample_point_y.npy')
# t_pde = np.load('resample_point_t.npy')

# convert pytorch Variable
x_pde = Variable(torch.from_numpy(x_pde).float(), requires_grad=True).to(device)
y_pde = Variable(torch.from_numpy(y_pde).float(), requires_grad=True).to(device)
t_pde = Variable(torch.from_numpy(t_pde).float(), requires_grad=True).to(device)

# iterations = 6*10**5
iterations = 3*10**5
min_loss = 10000000
time1 = time.time()
print(time1)
scaler = GradScaler()
for epoch in range(iterations):
    optimizer.zero_grad()  # to make the gradients zero
    with autocast():
        # Loss based on boundary conditions
        predict_bc = net(torch.cat([x_bc, y_bc, t_bc],1))
        predict_u_bc, predict_v_bc, predict_p_bc = predict_bc[:, 0].reshape(-1, 1), predict_bc[:, 1].reshape(-1, 1), predict_bc[:, 2].reshape(-1, 1)
        mse_u_bc = mse_cost_function(predict_u_bc, u_bc)
        mse_v_bc = mse_cost_function(predict_v_bc, v_bc)
        mse_p_bc = mse_cost_function(predict_p_bc, p_bc)
        #
        # # Loss based on initial value
        predict_ic = net(torch.cat([x_ic, y_ic, t_ic], 1))
        predict_u_ic, predict_v_ic, predict_p_ic = predict_ic[:, 0].reshape(-1, 1), predict_ic[:, 1].reshape(-1, 1), predict_ic[:, 2].reshape(-1, 1)
        mse_u_ic = mse_cost_function(predict_u_ic, u_ic)
        mse_v_ic = mse_cost_function(predict_v_ic, v_ic)
        mse_p_ic = mse_cost_function(predict_p_ic, p_ic)

        # Loss based on PDE
        f_u, f_v, f_e = NS_cylinder(x_pde, y_pde, t_pde, net)  # output of f(t,x)
        mse_f1 = mse_cost_function(f_u, torch.zeros_like(f_u))
        mse_f2 = mse_cost_function(f_v, torch.zeros_like(f_v))
        mse_f3 = mse_cost_function(f_e, torch.zeros_like(f_e))

        # Combining the loss functions
        loss = mse_f1 + mse_f2 + mse_f3 + mse_u_bc + mse_v_bc + mse_p_bc + mse_u_ic + mse_v_ic + mse_p_ic
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    # loss.backward()  # This is for computing gradients using backward propagation
    # optimizer.step()  # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta
    # scheduler.step(loss)

    if loss < min_loss:
        min_loss = loss
        # 保存模型语句
        # torch.save(net, '/home/user/ZHOU-Wen/PINN/demo/cylinder_wake/best_net_with_improved(DE).pth')
        print('current epoch',epoch)
    # with torch.autograd.no_grad():
    #     if epoch%1==0:
    #         data = [loss.item(), mse_f1.item(), mse_f2.item(), mse_f3.item(), mse_u_bc.item(), mse_v_bc.item(), mse_p_bc.item(), mse_u_ic.item(), mse_v_ic.item(), mse_p_ic.item()]
    #         with open('loss_no.csv', 'a', encoding='utf-8', newline='') as fp:
    #             # 写
    #             writer = csv.writer(fp)
    #             # 将数据写入
    #             writer.writerow(list(data))
    if epoch%1000==0:
        print(epoch, "Traning Loss:", loss.item())

time2 = time.time()
print(time2)
print('userd: {:.5f}s'.format(time2-time1))

