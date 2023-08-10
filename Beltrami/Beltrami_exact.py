import pandas as pd
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import datetime
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

mesh_x = 31
mesh_y = 31
x = np.linspace(-1,1,mesh_x)
print(x)
y = np.linspace(-1,1,mesh_y)
ms_x, ms_y = np.meshgrid(x, y)
x = np.ravel(ms_x).reshape(-1, 1)
y = np.ravel(ms_y).reshape(-1, 1)
z = 0.5 * np.ones((mesh_x*mesh_y, 1))
t = 1.0 * np.ones((mesh_x*mesh_y, 1))

u, v, w, p = data_generate(x,y,z,t)
temp = v
temp = np.array(temp).reshape(mesh_y,mesh_x)
temp = np.flipud(temp)
# Create a dataset
df = pd.DataFrame(temp)
print(df)

# Default heatmap
fig = plt.figure(figsize=(12,9))
plt.title("Exact v(x,y,z)",size=25)
ax = sns.heatmap(data=df,cmap="rainbow",robust=True,xticklabels=False,yticklabels=False)
ax.set_xticks(range(0, 31, 2))
ax.set_xticklabels(f'{c:.2f}' for c in np.arange(-1.0, 1.0, 2/16))
ax.set_yticks(range(0, 31, 2))
ax.set_yticklabels(f'{c:.2f}' for c in np.arange(1.0, -1.0, -2/16))
plt.yticks(size=10,)
plt.xticks(size=10,)
plt.show()