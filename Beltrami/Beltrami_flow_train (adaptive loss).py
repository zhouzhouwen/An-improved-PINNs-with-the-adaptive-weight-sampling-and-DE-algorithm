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

def NS_Beltrami(x,y,z,t):

    predict = net(torch.cat([x, y, z, t], 1))
    # reshape is very important and necessary
    u, v, w, p = predict[:,0].reshape(-1, 1), predict[:,1].reshape(-1, 1), predict[:,2].reshape(-1, 1), predict[:,3].reshape(-1, 1)

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

    f_u = u_t + (u * u_x + v * u_y + w * u_z) + p_x - 1 / Re * (u_xx + u_yy + u_zz)
    f_v = v_t + (u * v_x + v * v_y + w * v_z) + p_y - 1 / Re * (v_xx + v_yy + v_zz)
    f_w = w_t + (u * w_x + v * w_y + w * w_z) + p_z - 1 / Re * (w_xx + w_yy + w_zz)
    f_e = u_x + v_y + w_z

    return f_u, f_v, f_w, f_e

def gradients(y, x, order=1):
    if order == 1:
        return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),create_graph=True,only_inputs=True, )[0]
    else:
        return gradients(gradients(y, x), x, order=order - 1)

def data_generate(x, y, z, t):
    u = - a * (np.exp(a * x) * np.sin(a * y + d * z) + np.exp(a * z) * np.cos(a * x + d * y)) * np.exp(- d * d * t)
    v = - a * (np.exp(a * y) * np.sin(a * z + d * x) + np.exp(a * x) * np.cos(a * y + d * z)) * np.exp(- d * d * t)
    w = - a * (np.exp(a * z) * np.sin(a * x + d * y) + np.exp(a * y) * np.cos(a * z + d * x)) * np.exp(- d * d * t)
    p = - 0.5 * a * a * (np.exp(2 * a * x) + np.exp(2 * a * y) + np.exp(2 * a * z) +
                         2 * np.sin(a * x + d * y) * np.cos(a * z + d * x) * np.exp(a * (y + z)) +
                         2 * np.sin(a * y + d * z) * np.cos(a * x + d * y) * np.exp(a * (z + x)) +
                         2 * np.sin(a * z + d * x) * np.cos(a * y + d * z) * np.exp(a * (x + y))) * np.exp(
        -2 * d * d * t)

    return u, v, w, p

# if (os.path.exists('loss_no.csv')):
#     # 存在，则删除文件
#     os.remove('loss_no.csv')
# header = ['loss','mse_f1','mse_f2','mse_f3' ,'mse_f4', 'mse_u_bc' ,'mse_v_bc' ,'mse_w_bc' ,'mse_p_bc' ,'mse_u_ic' ,'mse_v_ic' ,'mse_w_ic' ,'mse_p_ic']
# with open('loss_no.csv', 'a', encoding='utf-8', newline='') as fp:
#     writer = csv.writer(fp)
#     writer.writerow(header)

device = torch.device("cuda")	# 使用gpu训练

net=Net(8, 100).to(device)
# net.apply(weights_init).to(device)
mse_cost_function = torch.nn.MSELoss(reduction='mean') # Mean squared error
optimizer = torch.optim.Adam(net.parameters(),)

Re = 1
a, d = 1, 1

# BC
data_bc = np.load('data_bc.npy')
x_bc = Variable(torch.from_numpy(data_bc[:, 0].reshape(-1, 1)).float(), requires_grad=False).to(device)
y_bc = Variable(torch.from_numpy(data_bc[:, 1].reshape(-1, 1)).float(), requires_grad=False).to(device)
z_bc = Variable(torch.from_numpy(data_bc[:, 2].reshape(-1, 1)).float(), requires_grad=False).to(device)
t_bc = Variable(torch.from_numpy(data_bc[:, 3].reshape(-1, 1)).float(), requires_grad=False).to(device)
u_bc = Variable(torch.from_numpy(data_bc[:, 4].reshape(-1, 1)).float(), requires_grad=False).to(device)
v_bc = Variable(torch.from_numpy(data_bc[:, 5].reshape(-1, 1)).float(), requires_grad=False).to(device)
w_bc = Variable(torch.from_numpy(data_bc[:, 6].reshape(-1, 1)).float(), requires_grad=False).to(device)
p_bc = Variable(torch.from_numpy(data_bc[:, 7].reshape(-1, 1)).float(), requires_grad=False).to(device)

# IC
data_ic = np.load('data_ic.npy')
x_ic = Variable(torch.from_numpy(data_ic[:, 0].reshape(-1, 1)).float(), requires_grad=False).to(device)
y_ic = Variable(torch.from_numpy(data_ic[:, 1].reshape(-1, 1)).float(), requires_grad=False).to(device)
z_ic = Variable(torch.from_numpy(data_ic[:, 2].reshape(-1, 1)).float(), requires_grad=False).to(device)
t_ic = Variable(torch.from_numpy(data_ic[:, 3].reshape(-1, 1)).float(), requires_grad=False).to(device)
u_ic = Variable(torch.from_numpy(data_ic[:, 4].reshape(-1, 1)).float(), requires_grad=False).to(device)
v_ic = Variable(torch.from_numpy(data_ic[:, 5].reshape(-1, 1)).float(), requires_grad=False).to(device)
w_ic = Variable(torch.from_numpy(data_ic[:, 6].reshape(-1, 1)).float(), requires_grad=False).to(device)
p_ic = Variable(torch.from_numpy(data_ic[:, 7].reshape(-1, 1)).float(), requires_grad=False).to(device)

# PDE

x_pde = np.random.random(5500)*2 -1 #用于生成5500个0到1的随机符点数: 0 <= n < 1.0
x_pde = np.array(x_pde).reshape(-1, 1)
y_pde = np.random.random(5500)*2 -1
y_pde = np.array(y_pde).reshape(-1, 1)
z_pde = np.random.random(5500)*2 -1
z_pde = np.array(z_pde).reshape(-1, 1)

t_pde = []
for j in range(11):
    t = 1 / 10 * j * np.ones(500).reshape(-1, 1)
    t_pde.append(t)
t_pde = np.array(t_pde)
t_pde = t_pde.reshape(-1, 1)
# pt_all_zeros = np.zeros((3000, 1))

# convert pytorch Variable
x_pde = Variable(torch.from_numpy(x_pde).float(), requires_grad=True).to(device)
y_pde = Variable(torch.from_numpy(y_pde).float(), requires_grad=True).to(device)
z_pde = Variable(torch.from_numpy(z_pde).float(), requires_grad=True).to(device)
t_pde = Variable(torch.from_numpy(t_pde).float(), requires_grad=True).to(device)
# pt_all_zeros = Variable(torch.from_numpy(pt_all_zeros).float(), requires_grad=False).to(device)

log_var_a = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
log_var_b = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
log_var_c = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
log_var_d = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
log_var_e = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
log_var_f = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
log_var_g = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
log_var_h = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
log_var_i = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
log_var_j = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
log_var_k = Variable(1 * torch.ones(1).cuda(), requires_grad=True)
log_var_l = Variable(1 * torch.ones(1).cuda(), requires_grad=True)

# get all parameters (model parameters + task dependent log variances)
params = ([p for p in net.parameters()] + [log_var_a] + [log_var_b] + [log_var_c] + [log_var_d] + [log_var_e] + [
    log_var_f] + [log_var_g] + [log_var_h] + [log_var_i] + [log_var_j] + [log_var_k] + [log_var_l])

mse_cost_function = torch.nn.MSELoss(reduction='mean')  # Mean squared error
optimizer = torch.optim.Adam(params, lr=0.001)
# optimizer = optim.Adahessian(params, lr= 1.0,betas= (0.9, 0.999),eps= 1e-4,weight_decay=0.0,hessian_power=1.0,)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1000,
                                                           verbose=True, threshold=0.0001, threshold_mode='rel',
                                                           cooldown=0, min_lr=0, eps=1e-08)
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

iterations = 3 * 10 ** 4
min_loss = 10000000
time1 = time.time()
print(time1)

for epoch in range(iterations):
    optimizer.zero_grad()  # to make the gradients zero
    # Loss based on boundary conditions
    predict_bc = net(torch.cat([x_bc, y_bc, z_bc, t_bc], 1))
    predict_u_bc, predict_v_bc, predict_w_bc, predict_p_bc = predict_bc[:, 0].reshape(-1, 1), predict_bc[:, 1].reshape(
        -1, 1), predict_bc[:, 2].reshape(-1, 1), predict_bc[:, 3].reshape(-1, 1)
    mse_u_bc = criterion(predict_u_bc, u_bc, log_var_a)
    mse_v_bc = criterion(predict_v_bc, v_bc, log_var_b)
    mse_w_bc = criterion(predict_w_bc, w_bc, log_var_c)
    mse_p_bc = criterion(predict_p_bc, p_bc, log_var_d)
    #
    # # Loss based on initial value
    predict_ic = net(torch.cat([x_ic, y_ic, z_ic, t_ic], 1))
    predict_u_ic, predict_v_ic, predict_w_ic, predict_p_ic = predict_ic[:, 0].reshape(-1, 1), predict_ic[:, 1].reshape(
        -1, 1), predict_ic[:, 2].reshape(-1, 1), predict_ic[:, 3].reshape(-1, 1)
    mse_u_ic = criterion(predict_u_ic, u_ic, log_var_e)
    mse_v_ic = criterion(predict_v_ic, v_ic, log_var_f)
    mse_w_ic = criterion(predict_w_ic, w_ic, log_var_g)
    mse_p_ic = criterion(predict_p_ic, p_ic, log_var_h)

    # Loss based on PDE
    f_u, f_v, f_w, f_e = NS_Beltrami(x_pde, y_pde, z_pde, t_pde,)  # output of f(t,x)
    mse_f1 = criterion(f_u, torch.zeros_like(f_u), log_var_i)
    mse_f2 = criterion(f_v, torch.zeros_like(f_v), log_var_j)
    mse_f3 = criterion(f_w, torch.zeros_like(f_w), log_var_k)
    mse_f4 = criterion(f_e, torch.zeros_like(f_e), log_var_l)

    # Combining the loss functions
    loss = mse_f1 + mse_f2 + mse_f3 + mse_f4 + mse_u_bc + mse_v_bc + mse_w_bc + mse_p_bc + mse_u_ic + mse_v_ic + mse_w_ic + mse_p_ic

    loss.backward()  # This is for computing gradients using backward propagation
    optimizer.step()  # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta
    # scheduler.step(loss)

    if loss < min_loss:
        min_loss = loss
        # 保存模型语句
        torch.save(net, '/home/user/ZHOU-Wen/PINN/demo/Beltrami/best_net_with_improved(adaptive loss).pth')
        print('current epoch:', epoch)

    # with torch.autograd.no_grad():
    #     if epoch%1==0:
    #         data = [loss.item(),mse_f1.item(), mse_f2.item(), mse_f3.item(), mse_f4.item(), mse_u_bc.item(), mse_v_bc.item(), mse_w_bc.item(), mse_p_bc.item(), mse_u_ic.item(), mse_v_ic.item(), mse_w_ic.item(), mse_p_ic.item()]
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


