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
x_pde = np.load('resample_point_x.npy')
y_pde = np.load('resample_point_y.npy')

plt.scatter(x_pde, y_pde)
plt.xlim(-0.5,1)
plt.ylim(-0.5,1.5)
plt.show()

