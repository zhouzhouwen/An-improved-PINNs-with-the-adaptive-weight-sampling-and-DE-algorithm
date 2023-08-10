# -*- coding: utf-8 -*-
"""MyProblem.py"""
from multiprocessing import Pool as ProcessPool
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool
import psutil
import geatpy as ea
import pandas as pd
import os
import subprocess
from scipy import integrate
import shutil
import csv
import random
import time
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from thop import profile
from torch.cuda.amp import autocast, GradScaler
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
from scoop import futures
from torch.utils.data import DataLoader, TensorDataset
import torch_optimizer as optim
from scipy.stats import qmc
import signal

"""
    该多目标优化BP神经网络框架是周文做出来的，版权所有。
"""

def NS_Beltrami(x, y, z, t, net):

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


# def N_match_S(data):
#     temp = []
#     for i in range(len(data)):
#         if data[i] == 1:
#             temp.append('nn.Softsign()')
#         if data[i] == 2:
#             temp.append('nn.Softplus()')
#         if data[i] == 3:
#             temp.append('nn.Tanh()')
#         if data[i] == 4:
#             temp.append('nn.Tanhshrink()')
#         if data[i] == 5:
#             temp.append('nn.ReLU()')
#         if data[i] == 6:
#             temp.append('nn.RReLU()')
#     return temp


def N_match_S(data):
    # temp = []
    # if data == 1:
    #     temp = 'nn.Softsign()'
    # if data == 2:
    #     temp = 'nn.Softplus()'
    if data == 1:
        temp = 'nn.Tanh()'
    # if data == 4:
    #     temp = 'nn.Tanhshrink()'
    if data == 2:
        temp = 'nn.ReLU()'
    # if data == 6:
    #     temp = 'nn.RReLU()'
    return temp

def mymovefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile,dstfile)          #移动文件
        print ("move %s -> %s"%( srcfile,dstfile))

def mycopyfile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.copyfile(srcfile,dstfile)      #复制文件
        print ("copy %s -> %s"%( srcfile,dstfile))
        
def del_file(filepath):
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath,f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


class MyProblem(ea.Problem):  # 继承Problem父类

    def __init__(self, PoolType):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        # Dim = 6+6+1  # 初始化Dim（决策变量维数）
        Dim = 4 # 初始化Dim（决策变量维数）
        # unit: m
        # self.var_set = np.arange(0.001, 0.278, common_difference)  # 设定一个集合，要求决策变量的值取自于该集合
        # self.var_set5 = np.arange(180, 200, 1)  # 设定一个集合，要求决策变量的值取自于该集合
        # self.var_set6 = np.arange(75, 95, 1)
        # self.var_set7 = np.arange(1, 7, 1)

        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        # 当你的变量是要去对应自己定义的数组时候，必须从0开始，因为numpy的第一个数据就是0
        lb = [0, 0, 0, 0]  # 决策变量下界
        ub = [9, 18, 4, 5]  # 决策变量上界
        # self.var_set_batch_size = np.array([8, 16, 32, 64, 128, 256])
        # self.var_set_init_lr = np.array([0.001, 0.01, 0.1, 1])
        lbin = [1] * Dim   # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim   # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)


        # 设置用多线程还是多进程
        self.PoolType = PoolType
        if self.PoolType == 'Thread':
            self.pool = ThreadPool(30)  # 设置池的大小
        elif self.PoolType == 'Process':
            num_cores = int(mp.cpu_count())  # 获得计算机的核心数
            print('cpu_count:',num_cores)
            # self.pool = ProcessPool(int(num_cores * 5/6))  # 设置池的大小
            self.pool = ProcessPool(int(6))  # 设置池的大小

    def evalVars(self, Vars):  # 目标函数，采用多线程加速计算
        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxx')
        N = Vars.shape[0]
        # args = list(zip(list(range(N)), [Vars] * N, [self.data] * N, [self.dataTarget] * N))
        args = list(zip([i for i in Vars], [False] * N))
        if self.PoolType == 'Thread':
            f = np.array(list(self.pool.map(subAimFunc, args)))
        elif self.PoolType == 'Process':
            result = self.pool.map_async(subAimFunc, args)
            result.wait()
            f = np.array(result.get())
            # result.close()

        # temp = []
        # pids = psutil.process_iter()
        # for pid in pids:
        #     if (pid.name() == 'python'):
        #         temp.append(pid.pid)
        # print(temp)
        # os.kill(pid, signal.SIGKILL)

        # pids = psutil.pids()
        # print(pids)
        # for pid in pids:
        #     p = psutil.Process(pid)
        #     # get process name according to pid
        #     process_name = p.name()
        #     # kill process "sleep_test1"
        #     if process_name == 'python':
        #         print("kill specific process: name(%s)-pid(%s)" % (process_name, pid))
        #         os.kill(pid, signal.SIGKILL)

        return f


def subAimFunc(args):

    def setup_seed(seed):
        torch.manual_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    setup_seed(3)

    device = torch.device("cuda")  # 使用gpu训练

    # BC
    data_bc = np.load('data_bc.npy')

    # global x_bc, y_bc, z_bc, t_bc, u_bc, v_bc, w_bc, p_bc
    # global x_ic, y_ic, z_ic, t_ic, u_ic, v_ic, w_ic, p_ic
    # global pt_x_collocation, pt_y_collocation, pt_z_collocation, pt_t_collocation

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

    x_pde = np.load('resample_point_x.npy')
    y_pde = np.load('resample_point_y.npy')
    z_pde = np.load('resample_point_z.npy')
    t_pde = np.load('resample_point_t.npy')

    # convert pytorch Variable
    pt_x_collocation = Variable(torch.from_numpy(x_pde).float(), requires_grad=True).to(device)
    pt_y_collocation = Variable(torch.from_numpy(y_pde).float(), requires_grad=True).to(device)
    pt_z_collocation = Variable(torch.from_numpy(z_pde).float(), requires_grad=True).to(device)
    pt_t_collocation = Variable(torch.from_numpy(t_pde).float(), requires_grad=True).to(device)

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


    phen, flag = args
    print('开始变量初始化')
    var_set_layer = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ])
    var_set_node_all_layer = np.array([30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 105, 120])
    var_set_init_lr = np.array([0.0001, 0.001, 0.01, 0.1, 1])
    var_set_act = np.array([0, 1, 2, 3, 4, 5])

    layer = var_set_layer[phen[0]]
    node = var_set_node_all_layer[phen[1]]
    init_lr = var_set_init_lr[phen[2]]
    act = var_set_act[phen[3]]

    class Net(nn.Module):
        def __init__(self, hidden_layers, nodes_per_layer, activation_function):
            input_node = 4
            output_node = 4
            super(Net, self).__init__()
            self.layers = nn.ModuleList()
            # 添加输入层到第一个隐藏层的连接
            self.layers.append(nn.Linear(input_node, nodes_per_layer))
            # 添加隐藏层
            for _ in range(hidden_layers):
                self.layers.append(nn.Linear(nodes_per_layer, nodes_per_layer))

            # 添加最后一层到输出层的连接
            self.layers.append(nn.Linear(nodes_per_layer, output_node))

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

    net = Net(layer, node, act)
    net.apply(weights_init).to(device)

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
    optimizer = torch.optim.Adam(params, lr=init_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=200,
                                                           verbose=False, threshold=0.0001, threshold_mode='rel',
                                                           cooldown=0, min_lr=0, eps=1e-08)

    min_loss = 1000000

    print('this children has began')
    time1 = time.time()
    print(time1)
    scaler = GradScaler()
    for epoch in range(30000):
        # torch.cuda.empty_cache()
        # if hasattr(torch.cuda, 'empty_cache'):
        #     torch.cuda.empty_cache()
        optimizer.zero_grad()  # to make the gradients zero
        with autocast():
            # Loss based on boundary conditions
            predict_bc = net(torch.cat([x_bc, y_bc, z_bc, t_bc], 1)).to(device)
            predict_u_bc, predict_v_bc, predict_w_bc, predict_p_bc = predict_bc[:, 0].reshape(-1, 1), predict_bc[:,1].reshape(-1,1), predict_bc[:,2].reshape(-1, 1), predict_bc[:, 3].reshape(-1, 1)
            mse_u_bc = criterion(predict_u_bc, u_bc, log_var_a)
            mse_v_bc = criterion(predict_v_bc, v_bc, log_var_b)
            mse_w_bc = criterion(predict_w_bc, w_bc, log_var_c)
            mse_p_bc = criterion(predict_p_bc, p_bc, log_var_d)
            #
            # # Loss based on initial value
            predict_ic = net(torch.cat([x_ic, y_ic, z_ic, t_ic], 1))
            predict_u_ic, predict_v_ic, predict_w_ic, predict_p_ic = predict_ic[:, 0].reshape(-1, 1), predict_ic[:,1].reshape(-1,1), predict_ic[:,2].reshape(-1, 1), predict_ic[:, 3].reshape(-1, 1)
            mse_u_ic = criterion(predict_u_ic, u_ic, log_var_e)
            mse_v_ic = criterion(predict_v_ic, v_ic, log_var_f)
            mse_w_ic = criterion(predict_w_ic, w_ic, log_var_j)
            mse_p_ic = criterion(predict_p_ic, p_ic, log_var_h)

            # Loss based on PDE
            f_u, f_v, f_w, f_e = NS_Beltrami(pt_x_collocation, pt_y_collocation, pt_z_collocation, pt_t_collocation, net=net)  # output of f(t,x)
            mse_f1 = criterion(f_u, torch.zeros_like(f_u), log_var_i)
            mse_f2 = criterion(f_v, torch.zeros_like(f_v), log_var_j)
            mse_f3 = criterion(f_w, torch.zeros_like(f_w), log_var_k)
            mse_f4 = criterion(f_e, torch.zeros_like(f_e), log_var_l)

            # Combining the loss functions
            loss = mse_f1 + mse_f2 + mse_f3 + mse_f4 + mse_u_bc + mse_v_bc + mse_w_bc + mse_p_bc + mse_u_ic + mse_v_ic + mse_w_ic + mse_p_ic
        scheduler.step(loss)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # loss.backward()  # This is for computing gradients using backward propagation
        # optimizer.step()  # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta


        if loss < min_loss:
            min_loss = loss
            # 保存模型语句
            # torch.save(net, '/home/user/ZHOU-Wen/PINN/demo/Kovasznay/best_net_with_improvedtt.pth')
            # print('saved')

    # train_error.append(min_loss)
    time2 = time.time()
    print(time2)
    print('userd: {:.5f}s'.format(time2 - time1))

    min_loss = min_loss.cpu().data.numpy()
    ObjV = np.hstack([min_loss])

    # pop.CV = np.hstack([
    #
    #
    #                     ]),

    # used in neural network for future
    # print('这一代结束啦')
    pd.DataFrame(np.hstack([np.array(phen).ravel(), ObjV.ravel()])).to_csv('all_var.csv', mode='a', index=False, header=None)


    return ObjV
