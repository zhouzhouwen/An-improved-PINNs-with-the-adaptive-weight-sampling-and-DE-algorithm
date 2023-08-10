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
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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


data = []
for i in range(11):
    sampler = qmc.Sobol(d=3, scramble=True)
    # sample = sampler.random_base2(m=10)
    sample = sampler.random(500)
    sample = sample * 2 - 1
    t = 1/10 * i * np.ones(500).reshape(-1,1)
    data.append(np.concatenate([sample,t],1))

data = np.array(data).reshape(-1,4)
print(data)
print(data.shape)

np.save('first_sample_point_sobol',data)
# print(data)
# print(data.shape)
# print(np.max(data))
# print(np.min(data))
data = data[0:550,0:3]

x=data[:,0]
y=data[:,1]
z=data[:,2]

# fig=plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x,y,z,c='r')
# plt.show()

#

# a = np.load('first_sample_point_sobol.npy')
# print(a)


x_pde = np.random.randint(31, size=500) / 15 - 1
x_pde = x_pde.reshape(-1, 1)
y_pde = np.random.randint(31, size=500) / 15 - 1
y_pde = y_pde.reshape(-1, 1)
z_pde = np.random.randint(31, size=500) / 15 - 1
z_pde = z_pde.reshape(-1, 1)
t_pde = np.random.randint(11, size=5500) / 10
t_pde = t_pde.reshape(-1, 1)

fig=plt.figure()
fig.add_subplot(121, projection='3d').scatter(x,y,z,c='r')
fig.add_subplot(122, projection='3d').scatter(x_pde,y_pde,z_pde,c='r')

plt.show()
