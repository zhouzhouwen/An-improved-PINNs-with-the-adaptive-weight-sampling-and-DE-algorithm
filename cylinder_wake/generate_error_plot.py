
import pandas as pd
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import time

data_no = np.load('error_no_improved.npy')
data_with = np.load('error_with_improved.npy')
print(data_no)
print(data_with)
t = np.linspace(0,19,20)/10
print(t)
fig = plt.figure()
plt.plot(t, data_no[:,2],marker='o',linestyle='dashed',label="Conventional PINN")
plt.plot(t, data_with[:,2],marker='o',linestyle='dashed',label="Improved PINN")
plt.legend()
plt.title('MAPE of p',size = 20)
plt.xlabel('Seconds')
plt.xticks(np.arange(0, 2.1, 0.2))
plt.show()