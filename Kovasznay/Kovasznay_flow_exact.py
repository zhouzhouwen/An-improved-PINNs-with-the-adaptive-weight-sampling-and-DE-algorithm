import pandas as pd
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import datetime

Re = 40
lam = 0.5 * Re - np.sqrt(0.25 * (Re ** 2) + 4 * (np.pi ** 2))

mesh_x = 300
mesh_y = 300

x = np.linspace(-0.5,1,mesh_x)
y = np.linspace(-0.5,1.5,mesh_y)
u = []
v = []
p = []
for i in range(len(y)):
    for j in range(len(x)):
        u.append(1 - np.exp(lam * x[j]) * np.cos(2 * np.pi * y[i]))
        v.append(lam / (2 * np.pi) * np.exp(lam * x[j]) * np.sin(2 * np.pi * y[i]))
        p.append(1 / 2 * (1 - np.exp(2 * lam * x[j])))


data = np.array(p).reshape(mesh_y,mesh_x)
data = np.flipud(data)
# Create a dataset
data = pd.DataFrame(data)
print(data)

# Default heatmap
fig = plt.figure()
plt.title("Exact p(x, y)",size=15)
ax = sns.heatmap(data=data,cmap="rainbow",robust=True,xticklabels=False,yticklabels=False)

ax.set_xticks(range(0, 300, 20))
ax.set_xticklabels(f'{c:.1f}' for c in np.arange(-0.5, 1.0, 0.1))
ax.set_yticks(range(0, 300, 15))
ax.set_yticklabels(f'{c:.1f}' for c in np.arange(1.5, -0.5, -0.1))

plt.show()