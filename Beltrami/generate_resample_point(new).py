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


# seed_everything(33)


# class Net(nn.Module):
#     def __init__(self, input_size=2, output_size=3):
#         super(Net, self).__init__()
#         self.layer = nn.Sequential(nn.Linear(input_size, 54),
#                                    # 字符型的变量后面不跟0，跟上一行区别
#                                    nn.Softsign(),
#                                    nn.Linear(54, 30),
#                                    nn.Tanhshrink(),
#                                    nn.Linear(30, 56),
#                                    nn.Tanhshrink(),
#                                    nn.Linear(56, 69),
#                                    nn.ReLU(),
#                                    nn.Linear(69, 57),
#                                    nn.Tanhshrink(),
#                                    nn.Linear(57, 61),
#                                    nn.Tanh(),
#                                    nn.Linear(61, output_size),
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
    def __init__(self, NL, NN):
        # NL是有多少层隐藏层
        # NN是每层的神经元数量
        super(Net, self).__init__()
        self.input_layer = nn.Linear(4, NN)
        self.hidden_layer = nn.ModuleList([nn.Linear(NN, NN) for i in range(NL)])
        self.output_layer = nn.Linear(NN, 4)

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


def NS_Beltrami(x, y, z, t, net):
    predict = net(torch.cat([x, y, z, t], 1))
    # reshape is very important and necessary
    u, v, w, p = predict[:, 0].reshape(-1, 1), predict[:, 1].reshape(-1, 1), predict[:, 2].reshape(-1, 1), predict[:,
                                                                                                           3].reshape(
        -1, 1)

    u_x = gradients(u, x)
    u_y = gradients(u, y)
    u_z = gradients(u, z)
    u_xx = gradients(u, x, 2)
    u_yy = gradients(u, y, 2)
    u_zz = gradients(u, z, 2)

    v_x = gradients(v, x)
    v_y = gradients(v, y)
    v_z = gradients(v, z)
    v_xx = gradients(v, x, 2)
    v_yy = gradients(v, y, 2)
    v_zz = gradients(v, z, 2)

    w_x = gradients(w, x)
    w_y = gradients(w, y)
    w_z = gradients(w, z)
    w_xx = gradients(w, x, 2)
    w_yy = gradients(w, y, 2)
    w_zz = gradients(w, z, 2)

    p_x = gradients(p, x)
    p_y = gradients(p, y)
    p_z = gradients(p, z)

    u_t = gradients(u, t)
    v_t = gradients(v, t)
    w_t = gradients(w, t)

    f_u = u_t + (u * u_x + v * u_y + w * u_z) + p_x - 1 / 1 * (u_xx + u_yy + u_zz)
    f_v = v_t + (u * v_x + v * v_y + w * v_z) + p_y - 1 / 1 * (v_xx + v_yy + v_zz)
    f_w = w_t + (u * w_x + v * w_y + w * w_z) + p_z - 1 / 1 * (w_xx + w_yy + w_zz)
    f_e = u_x + v_y + w_z

    return f_u, f_v, f_w, f_e


def gradients(y, x, order=1):
    if order == 1:
        return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True, )[0]
    else:
        return gradients(gradients(y, x), x, order=order - 1)


device = torch.device("cuda")  # 使用gpu训练

#
# def data_generate(x, y, z, t):
#     u = - a * (np.exp(a * x) * np.sin(a * y + d * z) + np.exp(a * z) * np.cos(a * x + d * y)) * np.exp(- d * d * t)
#     v = - a * (np.exp(a * y) * np.sin(a * z + d * x) + np.exp(a * x) * np.cos(a * y + d * z)) * np.exp(- d * d * t)
#     w = - a * (np.exp(a * z) * np.sin(a * x + d * y) + np.exp(a * y) * np.cos(a * z + d * x)) * np.exp(- d * d * t)
#     p = - 0.5 * a * a * (np.exp(2 * a * x) + np.exp(2 * a * y) + np.exp(2 * a * z) +
#                          2 * np.sin(a * x + d * y) * np.cos(a * z + d * x) * np.exp(a * (y + z)) +
#                          2 * np.sin(a * y + d * z) * np.cos(a * x + d * y) * np.exp(a * (z + x)) +
#                          2 * np.sin(a * z + d * x) * np.cos(a * y + d * z) * np.exp(a * (x + y))) * np.exp(
#         -2 * d * d * t)
#
#     return u, v, w, p
#
#
# net = Net(8, 100)
# net.apply(weights_init).to(device)
# Re = 1
# a, d = 1, 1
#
# # BC
# data_bc = np.load('data_bc.npy')
# x_bc = Variable(torch.from_numpy(data_bc[:, 0].reshape(-1, 1)).float(), requires_grad=False).to(device)
# y_bc = Variable(torch.from_numpy(data_bc[:, 1].reshape(-1, 1)).float(), requires_grad=False).to(device)
# z_bc = Variable(torch.from_numpy(data_bc[:, 2].reshape(-1, 1)).float(), requires_grad=False).to(device)
# t_bc = Variable(torch.from_numpy(data_bc[:, 3].reshape(-1, 1)).float(), requires_grad=False).to(device)
# u_bc = Variable(torch.from_numpy(data_bc[:, 4].reshape(-1, 1)).float(), requires_grad=False).to(device)
# v_bc = Variable(torch.from_numpy(data_bc[:, 5].reshape(-1, 1)).float(), requires_grad=False).to(device)
# w_bc = Variable(torch.from_numpy(data_bc[:, 6].reshape(-1, 1)).float(), requires_grad=False).to(device)
# p_bc = Variable(torch.from_numpy(data_bc[:, 7].reshape(-1, 1)).float(), requires_grad=False).to(device)
#
# # IC
# data_ic = np.load('data_ic.npy')
# x_ic = Variable(torch.from_numpy(data_ic[:, 0].reshape(-1, 1)).float(), requires_grad=False).to(device)
# y_ic = Variable(torch.from_numpy(data_ic[:, 1].reshape(-1, 1)).float(), requires_grad=False).to(device)
# z_ic = Variable(torch.from_numpy(data_ic[:, 2].reshape(-1, 1)).float(), requires_grad=False).to(device)
# t_ic = Variable(torch.from_numpy(data_ic[:, 3].reshape(-1, 1)).float(), requires_grad=False).to(device)
# u_ic = Variable(torch.from_numpy(data_ic[:, 4].reshape(-1, 1)).float(), requires_grad=False).to(device)
# v_ic = Variable(torch.from_numpy(data_ic[:, 5].reshape(-1, 1)).float(), requires_grad=False).to(device)
# w_ic = Variable(torch.from_numpy(data_ic[:, 6].reshape(-1, 1)).float(), requires_grad=False).to(device)
# p_ic = Variable(torch.from_numpy(data_ic[:, 7].reshape(-1, 1)).float(), requires_grad=False).to(device)
#
# # PDE
#
# sample = np.load('first_sample_point_sobol.npy')
# x_pde = sample[:, 0].reshape(-1, 1)
# y_pde = sample[:, 1].reshape(-1, 1)
# z_pde = sample[:, 2].reshape(-1, 1)
# t_pde = sample[:, 3].reshape(-1, 1)
# x_pde = Variable(torch.from_numpy(x_pde).float(), requires_grad=True).to(device)
# y_pde = Variable(torch.from_numpy(y_pde).float(), requires_grad=True).to(device)
# z_pde = Variable(torch.from_numpy(z_pde).float(), requires_grad=True).to(device)
# t_pde = Variable(torch.from_numpy(t_pde).float(), requires_grad=True).to(device)
# print(x_pde.shape)
#
# # # Define task dependent log_variance
# log_var_a = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
# log_var_b = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
# log_var_c = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
# log_var_d = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
# log_var_e = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
# log_var_f = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
# log_var_g = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
# log_var_h = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
# log_var_i = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
# log_var_j = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
# log_var_k = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
# log_var_l = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
#
# # get all parameters (model parameters + task dependent log variances)
# params = ([p for p in net.parameters()] + [log_var_a] + [log_var_b] + [log_var_c] + [log_var_d] + [log_var_e] + [
#     log_var_f] + [log_var_g] + [log_var_h] + [log_var_i] + [log_var_j] + [log_var_k] + [log_var_l])
# #
#
# mse_cost_function = torch.nn.MSELoss(reduction='mean')  # Mean squared error
# optimizer = torch.optim.Adam(params, lr=0.001)
#
# # optimizer = optim.Adahessian(params, lr= 1.0,betas= (0.9, 0.999),eps= 1e-4,weight_decay=0.0,hessian_power=1.0,)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=200, verbose=False,
#                                                        threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
#                                                        eps=1e-08)
#
#
# # define loss criterion
# def criterion(y_pred, y_true, log_vars):
#     loss = 0
#     # method 1
#     # for i in range(len(y_pred)):
#     #     precision = torch.exp(-log_vars)
#     #     diff = (y_pred - y_true) ** 2.0
#     #     loss += torch.sum(precision * diff + log_vars, -1)
#     # method 2
#     # for i in range(len(y_pred)):
#     #     precision = 0.5 / (log_vars ** 2)
#     #     diff = (y_pred - y_true) ** 2.0
#     #     loss += torch.sum(precision * diff + torch.log(1 + log_vars ** 2), -1)
#
#     precision = 0.5 / (log_vars ** 2)
#     diff = (y_pred - y_true) ** 2.0
#     loss += torch.sum(precision * diff + torch.log(1 + log_vars ** 2), -1)
#
#     return torch.mean(loss)
#
#
# iterations = 3 * 10 ** 4
# min_loss = 10000000
# time1 = time.time()
# print(time1)
#
# for epoch in range(iterations):
#     optimizer.zero_grad()  # to make the gradients zero
#     # Loss based on boundary conditions
#     predict_bc = net(torch.cat([x_bc, y_bc, z_bc, t_bc], 1))
#     predict_u_bc, predict_v_bc, predict_w_bc, predict_p_bc = predict_bc[:, 0].reshape(-1, 1), predict_bc[:, 1].reshape(
#         -1, 1), predict_bc[:, 2].reshape(-1, 1), predict_bc[:, 3].reshape(-1, 1)
#     mse_u_bc = criterion(predict_u_bc, u_bc, log_var_a)
#     mse_v_bc = criterion(predict_v_bc, v_bc, log_var_b)
#     mse_w_bc = criterion(predict_w_bc, w_bc, log_var_c)
#     mse_p_bc = criterion(predict_p_bc, p_bc, log_var_e)
#     #
#     # # Loss based on initial value
#     predict_ic = net(torch.cat([x_ic, y_ic, z_ic, t_ic], 1))
#     predict_u_ic, predict_v_ic, predict_w_ic, predict_p_ic = predict_ic[:, 0].reshape(-1, 1), predict_ic[:, 1].reshape(
#         -1, 1), predict_ic[:, 2].reshape(-1, 1), predict_ic[:, 3].reshape(-1, 1)
#     mse_u_ic = criterion(predict_u_ic, u_ic, log_var_e)
#     mse_v_ic = criterion(predict_v_ic, v_ic, log_var_f)
#     mse_w_ic = criterion(predict_w_ic, w_ic, log_var_g)
#     mse_p_ic = criterion(predict_p_ic, p_ic, log_var_h)
#
#     # Loss based on PDE
#     f_u, f_v, f_w, f_e = NS_Beltrami(x_pde, y_pde, z_pde, t_pde, net=net)  # output of f(t,x)
#     mse_f1 = criterion(f_u, torch.zeros_like(f_u), log_var_i)
#     mse_f2 = criterion(f_v, torch.zeros_like(f_v), log_var_j)
#     mse_f3 = criterion(f_w, torch.zeros_like(f_w), log_var_k)
#     mse_f4 = criterion(f_e, torch.zeros_like(f_e), log_var_l)
#
#     # Combining the loss functions
#     loss = mse_f1 + mse_f2 + mse_f3 + mse_f4 + mse_u_bc + mse_v_bc + mse_w_bc + mse_p_bc + mse_u_ic + mse_v_ic + mse_w_ic + mse_p_ic
#
#     loss.backward()  # This is for computing gradients using backward propagation
#     optimizer.step()  # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta
#     # scheduler.step(loss)
#
#     if loss < min_loss:
#         min_loss = loss
#         # 保存模型语句
#         torch.save(net, '/home/user/ZHOU-Wen/PINN/demo/Beltrami/generate_resample_point.pth')
#         # print('saved')
#
#     if epoch % 1000 == 0:
#         print(epoch, "Traning Loss:", loss.item())
#
#
# time2 = time.time()
# print(time2)
# print('userd: {:.5f}s'.format(time2 - time1))


def reinforced_resampling_function(y):
    # 应用平方函数来强化概率分布
    y = np.power(y, 2.0)
    # 重新标准化数组以确保所有的概率之和为1
    y /= np.sum(y)
    # y = y[:, 0]
    return y


times_slice = 11
resample_num = int(500 / 2)
k = 0.5
c = 1
total_num = 3000


def resample_function(f_u, f_v, f_w, f_e):
    # print(len(X))
    # print(f_u.shape)
    data_ids = []
    # print(f_u)
    # print(f_u.shape)
    for i in range(times_slice):
        temp_f_u = f_u[i * total_num:(i + 1) * total_num]
        temp_f_v = f_v[i * total_num:(i + 1) * total_num]
        temp_f_w = f_w[i * total_num:(i + 1) * total_num]
        temp_f_e = f_e[i * total_num:(i + 1) * total_num]

        err_f_u = np.power(temp_f_u, k) / np.power(temp_f_u, k).mean() + c
        err_f_u_normalized = (err_f_u / sum(err_f_u))[:, 0]
        ids_u = np.random.choice(a=len(temp_f_u), size=resample_num, replace=False,p=reinforced_resampling_function(err_f_u_normalized)) + i * total_num

        err_f_v = np.power(temp_f_v, k) / np.power(temp_f_v, k).mean() + c
        err_f_v_normalized = (err_f_v / sum(err_f_v))[:, 0]
        ids_v = np.random.choice(a=len(temp_f_v), size=resample_num, replace=False,p=reinforced_resampling_function(err_f_v_normalized)) + i * total_num

        err_f_w = np.power(temp_f_w, k) / np.power(temp_f_w, k).mean() + c
        err_f_w_normalized = (err_f_w / sum(err_f_w))[:, 0]
        ids_w = np.random.choice(a=len(temp_f_w), size=resample_num, replace=False,p=reinforced_resampling_function(err_f_w_normalized)) + i * total_num

        err_f_e = np.power(temp_f_e, k) / np.power(temp_f_e, k).mean() + c
        err_f_e_normalized = (err_f_e / sum(err_f_e))[:, 0]
        ids_e = np.random.choice(a=len(temp_f_e), size=resample_num, replace=False,p=reinforced_resampling_function(err_f_e_normalized)) + i * total_num

        ids = np.concatenate([ids_u, ids_v, ids_w, ids_e])
        ids = np.random.choice(a=ids, size=resample_num, replace=False)

        # 加上另一半的Sobol采样，这里有可能出现重复采样
        sampler = qmc.Sobol(d=1, scramble=True)
        sample = sampler.integers(l_bounds=0, u_bounds=2999, n=250, endpoint=True) + i * total_num
        sample = sample.ravel()

        ids = np.concatenate([ids, sample])
        data_ids.append(ids)

    data_ids = np.array(data_ids).ravel()
    return data_ids


# temp_net = Net(6, 64).to(device)
temp_net = torch.load('/home/user/ZHOU-Wen/PINN/demo/Beltrami/generate_resample_point.pth')

data = []
for j in range(11):
    sampler = qmc.Sobol(d=3, scramble=True)
    # sample = sampler.random_base2(m=10)
    sample = sampler.random(3000)
    sample = sample * 2 - 1
    t = 1 / 10 * j * np.ones(3000).reshape(-1, 1)
    data.append(np.concatenate([sample, t], 1))

data = np.array(data).reshape(-1, 4)

X = data[:, 0].reshape(-1, 1)
Y = data[:, 1].reshape(-1, 1)
Z = data[:, 2].reshape(-1, 1)
T = data[:, 3].reshape(-1, 1)

x_pde = Variable(torch.from_numpy(X).float(), requires_grad=True).to(device)
y_pde = Variable(torch.from_numpy(Y).float(), requires_grad=True).to(device)
z_pde = Variable(torch.from_numpy(Z).float(), requires_grad=True).to(device)
t_pde = Variable(torch.from_numpy(T).float(), requires_grad=True).to(device)

f_u, f_v, f_w, f_e = NS_Beltrami(x_pde, y_pde, z_pde, t_pde, net=temp_net)

f_u = np.abs(f_u.cpu().data.numpy())
f_v = np.abs(f_v.cpu().data.numpy())
f_w = np.abs(f_w.cpu().data.numpy())
f_e = np.abs(f_e.cpu().data.numpy())

ids = resample_function(f_u, f_v, f_w, f_e)
# print(ids)
# print(ids.shape)

X_selected = X[ids]
Y_selected = Y[ids]
Z_selected = Z[ids]
T_selected = T[ids]

np.save('resample_point_x', X_selected)
np.save('resample_point_y', Y_selected)
np.save('resample_point_z', Z_selected)
np.save('resample_point_t', T_selected)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
# 使用scatter方法生成散点图
ax.scatter(X_selected[5000:5500], Y_selected[5000:5500], Z_selected[5000:5500])
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()