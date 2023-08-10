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


def seed_everything(seed=3):
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

if (os.path.exists('loss_with.csv')):
    # 存在，则删除文件
    os.remove('loss_with.csv')
header = ['mse_f1', 'mse_f2', 'mse_f3', 'mse_u_bc', 'mse_v_bc', 'mse_p_bc']
with open('loss_with.csv', 'a', encoding='utf-8', newline='') as fp:
    writer = csv.writer(fp)
    writer.writerow(header)

net = Net(6, 64)
net.apply(weights_init).to(device)
Re = 40
lam = 0.5 * Re - np.sqrt(0.25 * (Re ** 2) + 4 * (np.pi ** 2))

# BC
# x_bc指的是流域的四条边，其他同理
x_bc = np.linspace(-0.5, 1.0, 101)
y_bc = np.linspace(-0.5, 1.5, 101)
x_bc = np.concatenate([np.array([-0.5] * 100), np.array([1] * 100), x_bc[0:100], x_bc[1:101]], 0)
y_bc = np.concatenate([y_bc[1:101], y_bc[0:100], np.array([-0.5] * 100), np.array([1.5] * 100)], 0)
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
# x = np.linspace(-0.5, 1, 300)
# y = np.linspace(-0.5, 1.5, 300)
# ms_x, ms_y = np.meshgrid(x, y)
# x_pde = np.ravel(ms_x).reshape(-1, 1)
# y_pde = np.ravel(ms_y).reshape(-1, 1)

# x_collocation = np.random.uniform(low=-0.5, high=1.0, size=(sample, 1))
# t_collocation = np.random.uniform(low=-0.5, high=1.5, size=(sample, 1))

sampler = qmc.Sobol(d=2, scramble=True)
# sample = sampler.random_base2(m=10)
sample = sampler.random(300)

x_pde = sample[:,0] * 1.5 - 0.5
y_pde = sample[:,1] * 2 - 0.5
x_pde = x_pde.reshape(-1,1)
y_pde = y_pde.reshape(-1,1)

# convert pytorch Variable
pt_x_collocation = Variable(torch.from_numpy(x_pde).float(), requires_grad=True).to(device)
pt_y_collocation = Variable(torch.from_numpy(y_pde).float(), requires_grad=True).to(device)


# Define task dependent log_variance

log_var_a = Variable(torch.ones(1).cuda(), requires_grad=True)
log_var_b = Variable(torch.ones(1).cuda(), requires_grad=True)
# log_var_c = Variable(lor_var, requires_grad=True)
# log_var_d = Variable(lor_var, requires_grad=True)
# log_var_e = Variable(lor_var, requires_grad=True)
# log_var_f = Variable(lor_var, requires_grad=True)
print(log_var_a)
print(log_var_b)
# print(log_var_c)
# print(log_var_d)
# print(log_var_e)
# print(log_var_f)

# get all parameters (model parameters + task dependent log variances)
# params = ([p for p in net.parameters()] + [log_var_a] + [log_var_b] + [log_var_c] + [log_var_d] + [log_var_e] + [log_var_f])
params = ([p for p in net.parameters()] + [log_var_a] + [log_var_b])

mse_cost_function = torch.nn.MSELoss(reduction='mean')  # Mean squared error
optimizer = torch.optim.Adam(params)
optimizer2 = torch.optim.LBFGS(params)
# optimizer = optim.Adahessian(params, lr= 1.0,betas= (0.9, 0.999),eps= 1e-4,weight_decay=0.0,hessian_power=1.0,)

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

# define loss criterion
def criterion(y_pred, y_true, log_vars):
    loss = 0
    # method 1
    # precision = torch.exp(-log_vars)
    # diff = (y_pred - y_true) ** 2.0
    # loss += torch.sum(precision * diff + log_vars, -1)

    # method 2
    precision = 0.5 / (log_vars ** 2)
    diff = (y_pred - y_true) ** 2.0
    loss += torch.sum(precision * diff + torch.log(1 + log_vars ** 2), -1)
    return torch.mean(loss)

iterations = 2 * 10 ** 2
min_loss = 10000000
time1 = time.time()
print(time1)


for epoch in range(100):
    optimizer.zero_grad()  # to make the gradients zero
    # Loss based on boundary conditions
    predict = net(torch.cat([x_bc, y_bc], 1))  # output of u(t,x)
    u, v, p = predict[:, 0].reshape(-1, 1), predict[:, 1].reshape(-1, 1), predict[:, 2].reshape(-1, 1)
    mse_u_bc = criterion(u, u_bc, log_var_a)
    mse_v_bc = criterion(v, v_bc, log_var_a)
    mse_p_bc = criterion(p, p_bc, log_var_a)
    #
    # # Loss based on initial value
    # net_ini_out1 = net(torch.cat([ pt_t_ini,pt_x_ini],1)) # output of u(t,x)
    # net_ini_out2 = u_t_grad(torch.cat([ pt_t_ini,pt_x_ini],1))
    # mse_u3 = mse_cost_function(net_ini_out1, pt_u_ini)
    # mse_u4 = mse_cost_function(net_ini_out2, pt_u_ini)

    # Loss based on PDE
    f_u, f_v, f_e = NS_Kovasznay(pt_x_collocation, pt_y_collocation)  # output of f(t,x)
    mse_f1 = criterion(f_u, torch.zeros_like(f_u), log_var_b)
    mse_f2 = criterion(f_v, torch.zeros_like(f_v), log_var_b)
    mse_f3 = criterion(f_e, torch.zeros_like(f_e), log_var_b)

    # Combining the loss functions
    loss = mse_f1 + mse_f2 + mse_f3 + mse_u_bc + mse_v_bc + mse_p_bc

    loss.backward()  # This is for computing gradients using backward propagation
    optimizer.step()  # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta
    # scheduler.step(loss)

    if loss < min_loss:
        min_loss = loss
        # 保存模型语句
        torch.save(net, '/home/user/ZHOU-Wen/PINN/demo/Kovasznay/best_net_with_improvedt.pth')
    with torch.autograd.no_grad():
        if epoch % 100 == 0:
            data = [mse_cost_function(f_u, torch.zeros_like(f_u)).item(), mse_cost_function(f_v, torch.zeros_like(f_v)).item(),
                    mse_cost_function(f_e, torch.zeros_like(f_e)).item(),
                    mse_cost_function(u, u_bc).item(), mse_cost_function(v, v_bc).item(),
                    mse_cost_function(p, p_bc).item(),
                    ]
            with open('loss_with.csv', 'a', encoding='utf-8', newline='') as fp:
                # 写
                writer = csv.writer(fp)
                # 将数据写入
                writer.writerow(list(data))
        if epoch % 100 == 0:
            print(epoch, "Traning Loss:", loss.data)
            print(epoch, "Traning Loss:", loss.item())

            # print(log_var_c)
            # print(log_var_d)
            # print(log_var_e)
            # print(log_var_f)

for epoch2 in range(100):

    def closure():
        optimizer.zero_grad()  # to make the gradients zero

        # Loss based on boundary conditions
        predict = net(torch.cat([x_bc, y_bc], 1))  # output of u(t,x)
        u, v, p = predict[:, 0].reshape(-1, 1), predict[:, 1].reshape(-1, 1), predict[:, 2].reshape(-1, 1)
        mse_u_bc = criterion(u, u_bc, log_var_a)
        mse_v_bc = criterion(v, v_bc, log_var_a)
        mse_p_bc = criterion(p, p_bc, log_var_a)
        #
        # # Loss based on initial value
        # net_ini_out1 = net(torch.cat([ pt_t_ini,pt_x_ini],1)) # output of u(t,x)
        # net_ini_out2 = u_t_grad(torch.cat([ pt_t_ini,pt_x_ini],1))
        # mse_u3 = mse_cost_function(net_ini_out1, pt_u_ini)
        # mse_u4 = mse_cost_function(net_ini_out2, pt_u_ini)

        # Loss based on PDE
        f_u, f_v, f_e = NS_Kovasznay(pt_x_collocation, pt_y_collocation)  # output of f(t,x)
        mse_f1 = criterion(f_u, torch.zeros_like(f_u), log_var_b)
        mse_f2 = criterion(f_v, torch.zeros_like(f_v), log_var_b)
        mse_f3 = criterion(f_e, torch.zeros_like(f_e), log_var_b)
        # Combining the loss functions
        loss2 = mse_f1 + mse_f2 + mse_f3 + mse_u_bc + mse_v_bc + mse_p_bc
        loss2 = loss2.to(device)
        loss2.backward()  # This is for computing gradients using backward propagation
        return loss2

    loss2 = closure()
    optimizer2.step(closure)  # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta
    # scheduler.step(loss)

    if loss2 < min_loss:
        min_loss = loss2
        # 保存模型语句
        torch.save(net, '/home/user/ZHOU-Wen/PINN/demo/Kovasznay/best_net_with_improvedt.pth')

    with torch.autograd.no_grad():
        if epoch2 % 100 == 0:
            data = [mse_cost_function(f_u, torch.zeros_like(f_u)).item(), mse_cost_function(f_v, torch.zeros_like(f_v)).item(),
                    mse_cost_function(f_e, torch.zeros_like(f_e)).item(),
                    mse_cost_function(u, u_bc).item(), mse_cost_function(v, v_bc).item(),
                    mse_cost_function(p, p_bc).item(),
                    ]
            with open('loss_with.csv', 'a', encoding='utf-8', newline='') as fp:
                # 写
                writer = csv.writer(fp)
                # 将数据写入
                writer.writerow(list(data))

        if epoch2 % 1 == 0:
            print(epoch2, "Traning Loss:", loss2.data)
            print(epoch2, "Traning Loss:", loss2.item())

time2 = time.time()
print(time2)
print('userd: {:.5f}s'.format(time2 - time1))


resample_num = 2000
k = 1
c = 1
for i in range(100):

    sampler = qmc.Sobol(d=2, scramble=True)
    # sample = sampler.random_base2(m=10)
    sample = sampler.random(1000)

    X = sample[:, 0] * 1.5 - 0.5
    Y = sample[:, 1] * 2 - 0.5
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)


    x_pde = Variable(torch.from_numpy(X).float(), requires_grad=True).to(device)
    y_pde = Variable(torch.from_numpy(Y).float(), requires_grad=True).to(device)


    net = torch.load('/home/user/ZHOU-Wen/PINN/demo/Kovasznay/best_net_with_improvedt.pth')

    f_u, f_v, f_e = NS_Kovasznay(x_pde, y_pde)

    f_u = np.abs(f_u.cpu().data.numpy())
    f_v = np.abs(f_v.cpu().data.numpy())
    f_e = np.abs(f_e.cpu().data.numpy())

    err = f_u + f_v + f_e

    err_eq = np.power(err, k) / np.power(err, k).mean() + c
    err_eq_normalized = (err_eq / sum(err_eq))[:, 0]

    ids = np.random.choice(a=len(x_pde), size=resample_num, replace=False, p=err_eq_normalized)
    X_selected = X[ids]
    Y_selected = Y[ids]

    x_pde = Variable(torch.from_numpy(X_selected).float(), requires_grad=True).to(device)
    y_pde = Variable(torch.from_numpy(Y_selected).float(), requires_grad=True).to(device)

    for epoch in range(100):
        optimizer.zero_grad()  # to make the gradients zero
        # Loss based on boundary conditions
        predict = net(torch.cat([x_bc, y_bc], 1))  # output of u(t,x)
        u, v, p = predict[:, 0].reshape(-1, 1), predict[:, 1].reshape(-1, 1), predict[:, 2].reshape(-1, 1)
        mse_u_bc = criterion(u, u_bc, log_var_a)
        mse_v_bc = criterion(v, v_bc, log_var_a)
        mse_p_bc = criterion(p, p_bc, log_var_a)
        #
        # # Loss based on initial value
        # net_ini_out1 = net(torch.cat([ pt_t_ini,pt_x_ini],1)) # output of u(t,x)
        # net_ini_out2 = u_t_grad(torch.cat([ pt_t_ini,pt_x_ini],1))
        # mse_u3 = mse_cost_function(net_ini_out1, pt_u_ini)
        # mse_u4 = mse_cost_function(net_ini_out2, pt_u_ini)

        # Loss based on PDE
        f_u, f_v, f_e = NS_Kovasznay(x_pde, y_pde)  # output of f(t,x)
        mse_f1 = criterion(f_u, torch.zeros_like(f_u), log_var_b)
        mse_f2 = criterion(f_v, torch.zeros_like(f_v), log_var_b)
        mse_f3 = criterion(f_e, torch.zeros_like(f_e), log_var_b)

        # Combining the loss functions
        loss = mse_f1 + mse_f2 + mse_f3 + mse_u_bc + mse_v_bc + mse_p_bc

        loss.backward()  # This is for computing gradients using backward propagation
        optimizer.step()  # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta
        # scheduler.step(loss)

        if loss < min_loss:
            min_loss = loss
            # 保存模型语句
            torch.save(net, '/home/user/ZHOU-Wen/PINN/demo/Kovasznay/best_net_with_improvedt.pth')
        with torch.autograd.no_grad():
            if epoch % 100 == 0:
                data = [mse_cost_function(f_u, torch.zeros_like(f_u)).item(), mse_cost_function(f_v, torch.zeros_like(f_v)).item(),
                        mse_cost_function(f_e, torch.zeros_like(f_e)).item(),
                        mse_cost_function(u, u_bc).item(), mse_cost_function(v, v_bc).item(),
                        mse_cost_function(p, p_bc).item(),
                        ]
                with open('loss_with.csv', 'a', encoding='utf-8', newline='') as fp:
                    # 写
                    writer = csv.writer(fp)
                    # 将数据写入
                    writer.writerow(list(data))
            if epoch % 100 == 0:
                print(epoch, "Traning Loss:", loss.data)
                print(epoch, "Traning Loss:", loss.item())

                # print(log_var_c)
                # print(log_var_d)
                # print(log_var_e)
                # print(log_var_f)

    for epoch2 in range(100):

        def closure():
            optimizer.zero_grad()  # to make the gradients zero

            # Loss based on boundary conditions
            predict = net(torch.cat([x_bc, y_bc], 1))  # output of u(t,x)
            u, v, p = predict[:, 0].reshape(-1, 1), predict[:, 1].reshape(-1, 1), predict[:, 2].reshape(-1, 1)
            mse_u_bc = criterion(u, u_bc, log_var_a)
            mse_v_bc = criterion(v, v_bc, log_var_a)
            mse_p_bc = criterion(p, p_bc, log_var_a)
            #
            # # Loss based on initial value
            # net_ini_out1 = net(torch.cat([ pt_t_ini,pt_x_ini],1)) # output of u(t,x)
            # net_ini_out2 = u_t_grad(torch.cat([ pt_t_ini,pt_x_ini],1))
            # mse_u3 = mse_cost_function(net_ini_out1, pt_u_ini)
            # mse_u4 = mse_cost_function(net_ini_out2, pt_u_ini)

            # Loss based on PDE
            f_u, f_v, f_e = NS_Kovasznay(x_pde, y_pde)  # output of f(t,x)
            mse_f1 = criterion(f_u, torch.zeros_like(f_u), log_var_b)
            mse_f2 = criterion(f_v, torch.zeros_like(f_v), log_var_b)
            mse_f3 = criterion(f_e, torch.zeros_like(f_e), log_var_b)
            # Combining the loss functions
            loss2 = mse_f1 + mse_f2 + mse_f3 + mse_u_bc + mse_v_bc + mse_p_bc
            loss2 = loss2.to(device)
            loss2.backward()  # This is for computing gradients using backward propagation
            return loss2


        loss2 = closure()
        optimizer2.step(closure)  # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta
        # scheduler.step(loss)

        if loss2 < min_loss:
            min_loss = loss2
            # 保存模型语句
            torch.save(net, '/home/user/ZHOU-Wen/PINN/demo/Kovasznay/best_net_with_improvedt.pth')

        with torch.autograd.no_grad():
            if epoch2 % 100 == 0:
                data = [mse_cost_function(f_u, torch.zeros_like(f_u)).item(), mse_cost_function(f_v, torch.zeros_like(f_v)).item(),
                        mse_cost_function(f_e, torch.zeros_like(f_e)).item(),
                        mse_cost_function(u, u_bc).item(), mse_cost_function(v, v_bc).item(),
                        mse_cost_function(p, p_bc).item(),
                        ]
                with open('loss_with.csv', 'a', encoding='utf-8', newline='') as fp:
                    # 写
                    writer = csv.writer(fp)
                    # 将数据写入
                    writer.writerow(list(data))

            if epoch2 % 1 == 0:
                print(epoch2, "Traning Loss:", loss2.data)
                print(epoch2, "Traning Loss:", loss2.item())

print(log_var_a)
print(log_var_b)