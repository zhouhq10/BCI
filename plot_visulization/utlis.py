import scipy.io as scio
import numpy as np
from pathlib import Path
import os
import random

import torch as th
import torch.nn as nn
from torch.utils import data
from torch.autograd import Variable
import nn as nn_spd
from optimizers import MixOptimizer
import functional
from model import SharedNet, BatchNet


def loaddata(datafile_base, num_sub):
    X_train = []
    Y_train = []
    train_cov = []
    for i in range(1,num_sub):
        datafile = datafile_base + str(i)
        data = scio.loadmat(str(datafile))
        MI_train = data['smt'].astype(float)
        MI_train = MI_train.transpose(1,2,0)
        MI_train = np.nan_to_num(MI_train)
        for i in range(100):
            cov1 = np.corrcoef(MI_train[i])
            train_cov.append(cov1)

        x_train = np.array(train_cov)
        y_train = np.transpose(data['y_dec']-1).astype(float)
        y_train = np.nan_to_num(y_train)
    Y_train.append(y_train)
    X_train = torch.from_numpy(np.nan_to_num(np.array(x_train)))
    Y_train = torch.from_numpy(np.array(Y_train))
    X_train = X_train.reshape(-1, 62, 62)
    Y_train = Y_train.reshape(-1, 1).long()
    X_train = X_train.reshape(100*num_sub, 1, 62, 62).float()

    train_dataset = Data.TensorDataset(X_train, Y_train)
    train_loader = Data.DataLoader(
        dataset=train_dataset,      # torch TensorDataset format
        batch_size=32,      # mini batch size
        shuffle=True          
        )
    return train_loader

def mmdloss(share):
    allmean = func.BaryGeom(share.reshape(-1,1,50,50))
    mmd_loss = Variable(torch.randn(32,1,50,50),requires_grad=True)
    for i in range(32):
        mean = func.BaryGeom(share[i].reshape(-1,1,50,50))
        dis = func.dist_riemann(allmean, mean)
        mmd_loss = mmd_loss + dis
    return mmd_loss

def logmap(B, P):
    B = np.array(B)
    P = np.array(P)
    BP = np.power(B, 0.5)
    BN = np.power(B, -0.5)
    S  = BP * np.log(BN * P * BN) * BP
    return S

