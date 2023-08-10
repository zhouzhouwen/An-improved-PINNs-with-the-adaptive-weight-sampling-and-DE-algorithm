# -*- coding: utf-8 -*-
# @Author: Wen Zhou
# @Date:   2021-07-07 20:17:13


from MyProblem_train1 import MyProblem  # 导入自定义问题接口
# -*- coding: utf-8 -*-
"""MyProblem.py"""
import numpy as np
import geatpy as ea
import pandas as pd
import os

import random
import torch

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


"""
    该多目标优化BP神经网络框架是周文做出来的，版权所有。
"""

if __name__ == '__main__':
    if os.path.exists('all_var.csv'):
        os.remove('all_var.csv')
        pd.DataFrame().to_csv('all_var.csv', index=False)
    else:
        pd.DataFrame().to_csv('all_var.csv',index=False)
    """===============================实例化问题对象============================"""
    problem = MyProblem(PoolType='Process')  # 生成问题对象
    # problem = MyProblem()  # 生成问题对象
    """==================================算法设置==============================="""
    algorithm = ea.soea_DE_best_1_bin_templet(
        problem,
        ea.Population(Encoding='RI',NIND=10,),
        MAXGEN=10,  # 最大进化代数
        logTras=1)  # 表示每隔多少代记录一次日志信息，0表示不记录。
    # 求解
    res = ea.optimize(algorithm,
                      verbose=True,
                      drawing=1,
                      outputMsg=True,
                      drawLog=True,
                      saveFlag=True)
    print(res)
