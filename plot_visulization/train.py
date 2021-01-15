import scipy.io as scio
import numpy as np
from pathlib import Path
import os
import random

from utlis import *
import torch as th
import torch.nn as nn
from torch.utils import data
from torch.autograd import Variable
import nn as nn_spd
from optimizers import MixOptimizer
import functional as func
from model import SharedNet, RieTransNet, RieBatchNet


def BciNet(train, test):
    feature1 = []
    #main parameters
    threshold_reeig=1e-4 #threshold for ReEig layer
    epochs=50
    loss_fn = nn.CrossEntropyLoss()

    #initial validation accuracy
    loss_val,acc_val=[],[]
    y_true,y_pred=[],[]
    gen = test
    model.eval()
    for local_batch, local_labels in gen:
        local_batch = local_batch.float().cuda()
        local_labels = local_labels.long().squeeze().cuda()
        out = model(local_batch)[0]
        l = loss_fn(out, local_labels)
        predicted_labels=out.argmax(1)
        y_true.extend(list(local_labels.cpu().numpy())); y_pred.extend(list(predicted_labels.cpu().numpy()))
        acc,loss=(predicted_labels==local_labels).cpu().numpy().sum()/out.shape[0], l.cpu().data.numpy()
        loss_val.append(loss)
        acc_val.append(acc)
    acc_val = np.asarray(acc_val).mean()
    loss_val = np.asarray(loss_val).mean()
    print('Initial validation accuracy: '+str(100*acc_val)+'%')

    #training loop
    best_acc = 0.00

    for epoch in range(epochs):
        opti = MixOptimizer(model.parameters(),lr = 0.005)

        # train one epoch
        loss_train, acc_train = [], []
        model.train()
        for local_batch, local_labels in train:
            local_batch = local_batch.float().cuda()
            local_labels = local_labels.long().squeeze().cuda()
            opti.zero_grad()
            out = model(local_batch)[0]
            if epoch==49:
              feature1.append(model(local_batch)[1])
            l = loss_fn(out, local_labels)
            acc, loss = (out.argmax(1)==local_labels).cpu().numpy().sum()/out.shape[0], l.cpu().data.numpy()
            loss_train.append(loss)
            acc_train.append(acc)
            l.backward()
            opti.step()
        acc_train = np.asarray(acc_train).mean()
        loss_train = np.asarray(loss_train).mean()
        #print(local_labels)
        #print(out)
        print('Train acc: ' + str(100*acc_train) + '% at epoch ' + str(epoch + 1) + '/' + str(epochs))


        # validation
        feature2 = []
        loss_val,acc_val=[],[]
        y_true,y_pred=[],[]
        gen = test
        model.eval()
        for local_batch, local_labels in gen:
            local_batch = local_batch.float().cuda()
            local_labels = local_labels.long().squeeze().cuda()
            out = model(local_batch)[0]
            feature2.append(model(local_batch)[1])
            l = loss_fn(out, local_labels)
            predicted_labels=out.argmax(1)
            y_true.extend(list(local_labels.cpu().numpy())); y_pred.extend(list(predicted_labels.cpu().numpy()))
            acc,loss=(predicted_labels==local_labels).cpu().numpy().sum()/out.shape[0], l.cpu().data.numpy()
            loss_val.append(loss)
            acc_val.append(acc)
        acc_val = np.asarray(acc_val).mean()
        loss_val = np.asarray(loss_val).mean()
        #print(local_labels)
        #print(out)
        #print(predicted_labels)
        if acc_val > best_acc:
            feature2most = feature2
            best_acc = acc_val
        print('Val acc: ' + str(100*acc_val) + '% at epoch ' + str(epoch + 1) + '/' + str(epochs))
        print(best_acc)
    return feature1, feature2most



def extract_feature(feature):
      feature_con = torch.cat([feature[0],feature[1],feature[2],feature[3]],dim=0)
      feature_mean = functional.BaryGeom(feature_con)
      logrie = []
      for i in range(100):
          a = np.array(feature_con[i][0].cpu().detach().numpy())
          b = logmap(a, feature_mean.cpu())
          c = b.reshape(50**2,-1)
          logrie.append(c)
      logrie = np.array(logrie).reshape(100,2500)
      mask = np.isnan(logrie)
      idx = np.where(~mask,np.arange(mask.shape[1]),0)
      logrie[mask] = logrie[np.nonzero(mask)[0], idx[mask]]   


feature_2 = torch.cat([feature2[0],feature2[1],feature2[2],feature2[3]],dim=0)

feature2_mean = functional.BaryGeom(feature_2)





logrie2 = []
for i in range(100):
  a = np.array(feature_2[i][0].cpu().detach().numpy())
  b = logmap(a, feature2_mean.cpu())
  c = b.reshape(50**2,-1)
  logrie2.append(c)
logrie2 = np.array(logrie2).reshape(100,2500)

mask = np.isnan(logrie2)
idx = np.where(~mask,np.arange(mask.shape[1]),0)
logrie2[mask] = logrie2[np.nonzero(mask)[0], idx[mask]]