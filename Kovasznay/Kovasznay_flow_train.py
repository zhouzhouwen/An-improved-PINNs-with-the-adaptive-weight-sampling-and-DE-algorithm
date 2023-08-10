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


def NS_Kovasznay(x, y):
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

    f_u = (u * u_x + v * u_y) + p_x - (1.0 / 40) * (u_xx + u_yy)
    f_v = (u * v_x + v * v_y) + p_y - (1.0 / 40) * (v_xx + v_yy)

    f_e = u_x + v_y

    return f_u, f_v, f_e


def gradients(y, x, order=1):
    if order == 1:
        return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True, )[0]
    else:
        return gradients(gradients(y, x), x, order=order - 1)


device = torch.device("cuda")  # 使用gpu训练

if (os.path.exists('loss_no.csv')):
    # 存在，则删除文件
    os.remove('loss_no.csv')
header = ['loss','mse_f1', 'mse_f2', 'mse_f3', 'mse_u_bc', 'mse_v_bc', 'mse_p_bc']
with open('loss_no.csv', 'a', encoding='utf-8', newline='') as fp:
    writer = csv.writer(fp)
    writer.writerow(header)

net = Net(8, 100)
net.apply(weights_init).to(device)
Re = 40
lam = 0.5 * Re - np.sqrt(0.25 * (Re ** 2) + 4 * (np.pi ** 2))

# BC
# x_bc指的是流域的四条边，其他同理
x_interval = 10
y_interval = 10
x_bc = np.linspace(-0.5, 1.0, x_interval+1)
y_bc = np.linspace(-0.5, 1.5, y_interval+1)
x_bc = np.concatenate([np.array([-0.5] * x_interval), np.array([1] * x_interval), x_bc[0:x_interval], x_bc[1:x_interval+1]], 0)
y_bc = np.concatenate([y_bc[1:y_interval+1], y_bc[0:y_interval], np.array([-0.5] * y_interval), np.array([1.5] * y_interval)], 0)
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
# x_pde = np.random.uniform(-0.5, 1, [100, 1])
# y_pde = np.random.uniform(-0.5, 1.5, [100, 1])

# x_pde = np.random.rand(100, 1) * 1.5 - 0.5
# y_pde = np.random.rand(100, 1) * 2 - 0.5
x_pde = np.random.uniform(low=-0.5, high=1.0, size=(100, 1))
y_pde = np.random.uniform(low=-0.5, high=1.5, size=(100, 1))
plt.scatter(x_pde, y_pde,)
plt.show()
# x = np.linspace(-0.5, 1, 101)
# y = np.linspace(-0.5, 1.5, 101)
# x, y = np.meshgrid(x, y)
# x_pde = np.ravel(x).reshape(-1, 1)
# y_pde = np.ravel(y).reshape(-1, 1)

pt_x_collocation = Variable(torch.from_numpy(x_pde).float(), requires_grad=True).to(device)
pt_y_collocation = Variable(torch.from_numpy(y_pde).float(), requires_grad=True).to(device)

iterations = 10000
min_loss = 10000000
time1 = time.time()
print(time1)

mse_cost_function = torch.nn.MSELoss(reduction='mean') # Mean squared error
optimizer = torch.optim.Adam(net.parameters())
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

for epoch in range(iterations):

    # Loss based on boundary conditions
    predict = net(torch.cat([x_bc, y_bc], 1))  # output of u(t,x)
    u, v, p = predict[:, 0].reshape(-1, 1), predict[:, 1].reshape(-1, 1), predict[:, 2].reshape(-1, 1)
    mse_u_bc = mse_cost_function(u, u_bc)
    mse_v_bc = mse_cost_function(v, v_bc)
    mse_p_bc = mse_cost_function(p, p_bc)
    #
    # Loss based on PDE
    f_u, f_v, f_e = NS_Kovasznay(pt_x_collocation, pt_y_collocation)  # output of f(t,x)

    mse_f1 = mse_cost_function(f_u, torch.zeros_like(f_u))
    mse_f2 = mse_cost_function(f_v, torch.zeros_like(f_v))
    mse_f3 = mse_cost_function(f_e, torch.zeros_like(f_e))

    # Combining the loss functions
    loss = mse_f1 + mse_f2 + mse_f3 + mse_u_bc + mse_v_bc + mse_p_bc

    optimizer.zero_grad()  # to make the gradients zero
    loss.backward()  # This is for computing gradients using backward propagation
    optimizer.step()  # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta
    # scheduler.step(loss)

    if loss < min_loss:
        min_loss = loss
        # 保存模型语句
        torch.save(net, '/home/user/ZHOU-Wen/PINN/demo/Kovasznay/best_net_no_improved.pth')
    with torch.autograd.no_grad():
        if epoch % 1 == 0:
            data = [loss.item(),mse_f1.item(), mse_f2.item(), mse_f3.item(), mse_u_bc.item(), mse_v_bc.item(), mse_p_bc.item()]
            with open('loss_no.csv', 'a', encoding='utf-8', newline='') as fp:
                # 写
                writer = csv.writer(fp)
                # 将数据写入
                writer.writerow(list(data))
        if epoch % 100 == 0:
            print(epoch, "Traning Loss:", loss.item())

time2 = time.time()
print(time2)
print('userd: {:.5f}s'.format(time2 - time1))
