import pandas as pd
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import time
import scipy.io
from scipy.interpolate import interp1d, interp2d
from scipy.interpolate import griddata

# read data
real_data = np.load('data_domain.npy')
real_data = real_data[real_data[:,2] == 1.0]

x = real_data[:, 0].reshape(-1,1)
y = real_data[:, 1].reshape(-1,1)
t = real_data[:, 2].reshape(-1,1)
real_u = real_data[:, 3]
real_v = real_data[:, 4]
real_p = real_data[:, 5]

temp = real_p.reshape(50,100)
# temp = np.flipud(temp)
# Create a dataset
df = pd.DataFrame(temp)

fig = plt.figure(figsize=(7, 4))
plt.title("Exact p(x,y)",size=15)
ax = sns.heatmap(data=df,cmap="rainbow",xticklabels=False,yticklabels=False)
ax.set_xticks([0,14,28,43,57,71,85,100])
ax.set_xticklabels(f'{c:.2f}' for c in [1,2,3,4,5,6,7,8])
ax.set_yticks([0,12,24,36,49])
ax.set_yticklabels(f'{c:.2f}' for c in [2,1,0,-1,-2])
plt.show()

