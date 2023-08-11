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

for i in range(1):
    sampler = qmc.Sobol(d=1, scramble=True)
    # sample = sampler.random_base2(m=10)
    sample = sampler.integers(l_bounds=0, u_bounds=99999, n=20000, endpoint=True)

print(sample)
print(sample.shape)
# np.save('first_sample_point_sobol',sample)
