# -*- coding: utf-8 -*-
"""MyProblem.py"""
from multiprocessing import Pool as ProcessPool
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool

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

Re = 1
a, d = 1, 1

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


x_mesh = 10
y_mesh = 10
z_mesh = 10

# BC
x1 = np.linspace(-1, 1, x_mesh+1)
print(x1)
print(x1.shape)
y1 = np.linspace(-1, 1, y_mesh+1)
z1 = np.linspace(-1, 1, z_mesh+1)
t1 = np.linspace(0, 1, 11)
b0 = np.array([-1] * 10 * 10)  # 30*30
b1 = np.array([1] * 10 * 10)  # 30*30

xt = np.tile(x1[0:x_mesh], x_mesh)  # 31-1
yt = np.tile(y1[0:y_mesh], y_mesh)
# zt = np.tile(z1[0:30], 30)
xt1 = np.tile(x1[1:x_mesh+1], x_mesh)
yt1 = np.tile(y1[1:y_mesh+1], y_mesh)
# zt1 = np.tile(z1[1:31], 30)

# xr = x1[0:30].repeat(30)
yr = y1[0:y_mesh].repeat(y_mesh)
zr = z1[0:z_mesh].repeat(z_mesh)
# xr1 = x1[1:31].repeat(30)
yr1 = y1[1:y_mesh+1].repeat(y_mesh)
zr1 = z1[1:z_mesh+1].repeat(z_mesh)

train1x = np.concatenate([b1, b0, xt1, xt, xt1, xt], 0).repeat(t1.shape[0])
train1y = np.concatenate([yt, yt1, b1, b0, yr1, yr], 0).repeat(t1.shape[0])
train1z = np.concatenate([zr, zr1, zr, zr1, b1, b0], 0).repeat(t1.shape[0])
train1t = np.tile(t1, x_mesh*x_mesh*6)  # The cube has six surfaces, 30*30*6

# train1x, train1y, train1z, form the six surfaces of a cube
train1ub, train1vb, train1wb, train1pb = data_generate(train1x, train1y, train1z, train1t)

x_bc = train1x.reshape(train1x.shape[0], 1)
y_bc = train1y.reshape(train1y.shape[0], 1)
z_bc = train1z.reshape(train1z.shape[0], 1)
t_bc = train1t.reshape(train1t.shape[0], 1)
u_bc = train1ub.reshape(train1ub.shape[0], 1)
v_bc = train1vb.reshape(train1vb.shape[0], 1)
w_bc = train1wb.reshape(train1wb.shape[0], 1)
p_bc = train1pb.reshape(train1pb.shape[0], 1)

data_bc = np.concatenate([x_bc, y_bc, z_bc, t_bc, u_bc, v_bc, w_bc, p_bc], 1)
print(data_bc.shape)

# IC
x_0 = np.tile(x1, (x_mesh+1) * (x_mesh+1))
y_0 = np.tile(y1.repeat(y_mesh+1), y_mesh+1)
z_0 = z1.repeat((z_mesh+1) * (z_mesh+1))
t_0 = np.array([0] * x_0.shape[0])
print(x_0.shape)
u_0, v_0, w_0, p_0 = data_generate(x_0, y_0, z_0, t_0)

x_ic = x_0.reshape(x_0.shape[0], 1)
y_ic = y_0.reshape(y_0.shape[0], 1)
z_ic = z_0.reshape(z_0.shape[0], 1)
t_ic = t_0.reshape(t_0.shape[0], 1)
u_ic = u_0.reshape(u_0.shape[0], 1)
v_ic = v_0.reshape(v_0.shape[0], 1)
w_ic = w_0.reshape(w_0.shape[0], 1)
p_ic = p_0.reshape(p_0.shape[0], 1)

data_ic = np.concatenate([x_ic, y_ic, z_ic, t_ic, u_ic, v_ic, w_ic, p_ic], 1)
print(data_ic.shape)
# np.save("data_bc",data_bc)
# np.save("data_ic",data_ic)