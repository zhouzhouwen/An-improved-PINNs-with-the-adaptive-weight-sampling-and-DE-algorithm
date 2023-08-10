# -*- coding: utf-8 -*-
"""MyProblem.py"""

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
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
from scoop import futures
from torch.utils.data import DataLoader, TensorDataset
import torch_optimizer as optim
from scipy.stats import qmc

"""
    该多目标优化BP神经网络框架是周文做出来的，版权所有。
"""




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

    f_u = (u * u_x + v * u_y) + p_x - (1.0 / 40) * (u_xx + u_yy)
    f_v = (u * v_x + v * v_y) + p_y - (1.0 / 40) * (v_xx + v_yy)

    f_e = u_x + v_y

    return f_u, f_v, f_e


def gradients(y, x, order=1):
    if order == 1:
        return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True, )[0]
    else:
        return gradients(gradients(y, x), x, order=order - 1)





def N_match_S(data):
    temp = []
    for i in range(len(data)):
        if data[i] == 1:
            temp.append('nn.Softsign()')
        if data[i] == 2:
            temp.append('nn.Softplus()')
        if data[i] == 3:
            temp.append('nn.Tanh()')
        if data[i] == 4:
            temp.append('nn.Tanhshrink()')
        if data[i] == 5:
            temp.append('nn.ReLU()')
        if data[i] == 6:
            temp.append('nn.RReLU()')
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

    def __init__(self):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 6+6+1  # 初始化Dim（决策变量维数）
        # unit: m
        # self.var_set = np.arange(0.001, 0.278, common_difference)  # 设定一个集合，要求决策变量的值取自于该集合
        # self.var_set5 = np.arange(180, 200, 1)  # 设定一个集合，要求决策变量的值取自于该集合
        # self.var_set6 = np.arange(75, 95, 1)
        # self.var_set7 = np.arange(1, 7, 1)

        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        # 当你的变量是要去对应自己定义的数组时候，必须从0开始，因为numpy的第一个数据就是0
        lb = [30, 30, 30, 30, 30, 30, 1, 1, 1, 1, 1, 1, 0]  # 决策变量下界
        ub = [100, 100, 100, 100, 100, 100, 6, 6, 6, 6, 6, 6, 3]  # 决策变量上界
        # self.var_set_batch_size = np.array([8, 16, 32, 64, 128, 256])
        self.var_set_init_lr = np.array([0.001, 0.01, 0.1, 1])
        lbin = [1] * Dim   # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim   # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def evalVars(self, pop):  # 目标函数，这里一共会运行MAXGEN次
        Vars = pop.Phen.astype(np.int32)  # 强制类型转换确保元素是整数
        # Vars  =  np.array([[5.0,204.0,293.0,4.0,3.0,4.0,4.0,2.0]]).astype(np.int32)
        # Vars = pop.Phen.astype(np.float)  # 强制类型转换确保元素是浮点数

        # 得到所有自变量列向量
        print('开始变量初始化')

        node1 = Vars[:, [0]]
        node2 = Vars[:, [1]]
        node3 = Vars[:, [2]]
        node4 = Vars[:, [3]]
        node5 = Vars[:, [4]]
        node6 = Vars[:, [5]]
        act1 = Vars[:, [6]]
        act2 = Vars[:, [7]]
        act3 = Vars[:, [8]]
        act4 = Vars[:, [9]]
        act5 = Vars[:, [10]]
        act6 = Vars[:, [11]]
        init_lr = self.var_set_init_lr[Vars[:, [12]]]

        act1 = N_match_S(act1)
        act2 = N_match_S(act2)
        act3 = N_match_S(act3)
        act4 = N_match_S(act4)
        act5 = N_match_S(act5)
        act6 = N_match_S(act6)

        train_error = []
        # val_error = []
        # flops = []

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

        # define constant values
        Re = 40
        lam = 0.5 * Re - np.sqrt(0.25 * (Re ** 2) + 4 * (np.pi ** 2))
        device = torch.device("cuda")  # 使用gpu训练

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

        sample = np.load('first_sample_point_sobol.npy')
        x_pde = sample[:, 0]
        y_pde = sample[:, 1]
        x_pde = x_pde.reshape(-1, 1)
        y_pde = y_pde.reshape(-1, 1)

        # convert pytorch Variable
        pt_x_collocation = Variable(torch.from_numpy(x_pde).float(), requires_grad=True).to(device)
        pt_y_collocation = Variable(torch.from_numpy(y_pde).float(), requires_grad=True).to(device)

        # Define task dependent log_variance
        log_var_a = Variable(torch.ones(1).cuda(), requires_grad=True)
        log_var_b = Variable(torch.ones(1).cuda(), requires_grad=True)

        # print(log_var_a)
        # print(log_var_b)

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








        for number in range(0,len(Vars)): #  the latter number means NIND in main.py

            class Net(nn.Module):
                def __init__(self, input_size = 2, output_size = 3):
                    super(Net, self).__init__()
                    self.layer = nn.Sequential(nn.Linear(input_size, node1[number, 0]),
                                               # 字符型的变量后面不跟0，跟上一行区别
                                               eval(act1[number]),
                                               nn.Linear(node1[number, 0], node2[number, 0]),
                                               eval(act2[number]),
                                               nn.Linear(node2[number, 0], node3[number, 0]),
                                               eval(act3[number]),
                                               nn.Linear(node3[number, 0], node4[number, 0]),
                                               eval(act4[number]),
                                               nn.Linear(node4[number, 0], node5[number, 0]),
                                               eval(act5[number]),
                                               nn.Linear(node5[number, 0], node6[number, 0]),
                                               eval(act6[number]),
                                               nn.Linear(node6[number, 0], output_size),
                                               )

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

            net = Net()
            net.apply(weights_init).to(device)

            # get all parameters (model parameters + task dependent log variances)
            # params = ([p for p in net.parameters()] + [log_var_a] + [log_var_b] + [log_var_c] + [log_var_d] + [log_var_e] + [log_var_f])
            params = ([p for p in net.parameters()] + [log_var_a] + [log_var_b])

            mse_cost_function = torch.nn.MSELoss(reduction='mean')  # Mean squared error
            optimizer = torch.optim.Adam(params, lr=init_lr[number,0])

            min_loss = 1000000

            print('this children has began')
            time1 = time.time()
            print(time1)
            for epoch in range(10000):
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
                f_u, f_v, f_e = NS_Kovasznay(pt_x_collocation, pt_y_collocation, net=net)  # output of f(t,x)
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
                    # torch.save(net, '/home/user/ZHOU-Wen/PINN/demo/Kovasznay/best_net_with_improvedtt.pth')
                    # print('saved')

            train_error.append(min_loss)
            time2 = time.time()
            print(time2)
            print('userd: {:.5f}s'.format(time2 - time1))

        train_error = np.array(train_error).reshape(-1, 1)
        pop.ObjV = np.hstack([train_error])

        # pop.CV = np.hstack([
        #
        #
        #                     ]),

        # used in neural network for future
        print('这一代结束啦')

        pd.DataFrame(np.hstack([Vars, train_error])).to_csv('all_var.csv', mode='a', index=False, header=None)



