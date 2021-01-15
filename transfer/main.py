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
import functional as func
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

# parameters


def mi_net(train, test):
    #main parameters
    model = BatchNet().double()
    lr=0.005
    loss_fn = nn.CrossEntropyLoss()
    threshold_reeig=1e-4 #threshold for ReEig layer
    epochs=100


    #training loop
    best_acc = 0.00
    for epoch in range(epochs):
        opti = MixOptimizer(model.parameters(),lr=lr)
        # train one epoch
        loss_train, acc_train = [], []
        model.train()
        for i,data in enumerate(zip(train, test)):
            source_batch=data[0][0].double()
            source_label=data[0][1]
            target_batch=data[1][0].double()
            source_label = source_label.squeeze()

            #opti1.zero_grad()
            opti.zero_grad()
            feature, out = model(source_batch, target_batch)

            feature = feature.double()
            out = out.double()
            #out = model(feature, target_batch)

            l1 = mmdloss(feature).double()
            l2 = loss_fn(out, source_label)
            l = l2#l1/32 + l2
            acc, loss = (out.argmax(1)==source_label).cpu().numpy().sum()/out.shape[0], l.cpu().data.numpy()
            loss_train.append(loss)
            acc_train.append(acc)
            l.sum().backward()
            opti.step()
        acc_train = np.asarray(acc_train).mean()
        loss_train = np.asarray(loss_train).mean()
        #print(local_labels)
        #print(out)
        print('Train acc: ' + str(100*acc_train) + '% at epoch ' + str(epoch + 1) + '/' + str(epochs))

        # validation
        loss_val,acc_val=[],[]
        y_true,y_pred=[],[]
        gen = test
        model.eval()
        for local_batch, local_labels in gen:
            local_labels = local_labels.squeeze()
            feature, out = model2(local_batch, local_batch)
            l = loss_fn(out, local_labels)
            predicted_labels=out.argmax(1)
            y_true.extend(list(local_labels.cpu().numpy())); 
            y_pred.extend(list(predicted_labels.cpu().detach().numpy()))
            acc,loss=(predicted_labels==local_labels).cpu().numpy().sum()/out.shape[0], l.cpu().data.numpy()
            loss_val.append(loss)
            acc_val.append(acc)
        acc_val = np.asarray(acc_val).mean()
        loss_val = np.asarray(loss_val).mean()
        #print(local_labels)
        #print(out)
        #print(predicted_labels)
        if acc_val > best_acc:
            best_acc = acc_val
        print('Val acc: ' + str(100*acc_val) + '% at epoch ' + str(epoch + 1) + '/' + str(epochs))
        print(best_acc)
