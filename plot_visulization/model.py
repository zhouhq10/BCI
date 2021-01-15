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

class SharedNet(nn.Module):
      def __init__(self):
          super(__class__,self).__init__()
          dim=62
          dim1=58; dim2=54; dim3=50
          self.re=nn_spd.ReEig()
          self.bimap1=nn_spd.BiMap(1,1,dim,dim1)
          self.bimap2=nn_spd.BiMap(1,1,dim1,dim2)
          self.bimap3=nn_spd.BiMap(1,1,dim2,dim3)
          self.logeig=nn_spd.LogEig()
          self.dropout1 = nn.Dropout(p=0.3)
          self.dropout2 = nn.Dropout(p=0.3)
      def forward(self,x):
          x_spd = self.bimap1(x)
          x_spd = self.re(x_spd)
          x_spd = self.dropout1(x_spd)
          x_spd = self.bimap2(x_spd)
          x_spd = self.re(x_spd)
          x_spd = self.dropout2(x_spd)
          x_spd = self.bimap3(x_spd)
          x_feature = self.re(x_spd)
          return x_feature.double()

class RieTransNet(nn.Module):
      def __init__(self):
          super(__class__,self).__init__()
          classes=2
          self.sharedNet=SharedNet().double()
          self.newbatch=nn_spd.NewBatchNormSPD(50).double()
          self.re=nn_spd.ReEig()
          self.logeig=nn_spd.LogEig()
          self.linear=nn.Linear(50**2,classes,bias=True)
          self.dropout1 = nn.Dropout(p=0.3)
          self.dropout2 = nn.Dropout(p=0.3)
          self.dropout3 = nn.Dropout(p=0.5)
          self.linear.weight.data.normal_(0, 0.005)
      def forward(self,source,target):
          #feature=self.featureNet(source)
          #target=self.targetNet(target)

          target_feature=self.sharedNet(target)
          source_feature=self.sharedNet(source)

          source=self.newbatch(source_feature,target_feature)
          source=self.logeig(source).view(source.shape[0],-1)
          source=self.dropout3(source)
          y=self.linear(source.double())
          return source_feature, y

class RieBatchNet(nn.Module):
      def __init__(self):
          super(__class__,self).__init__()
          dim=62
          dim1=58; dim2=54; dim3=50
          classes=2
          self.re=nn_spd.ReEig()
          self.bimap1=nn_spd.BiMap(1,1,dim,dim1)
          self.batchnorm1=nn_spd.BatchNormSPD(dim1)
          self.bimap2=nn_spd.BiMap(1,1,dim1,dim2)
          self.batchnorm2=nn_spd.BatchNormSPD(dim2)
          self.bimap3=nn_spd.BiMap(1,1,dim2,dim3)
          self.batchnorm3=nn_spd.BatchNormSPD(dim3)
          self.logeig=nn_spd.LogEig()
          self.linear=nn.Linear(dim3**2,classes,bias=True)
          self.dropout1 = nn.Dropout(p=0.3)
          self.dropout2 = nn.Dropout(p=0.3)
          self.dropout3 = nn.Dropout(p=0.5)
      def forward(self,x):
          x_spd=self.re(self.batchnorm1(self.bimap1(x)))
          #x_spd=self.dropout1(x_spd)
          x_spd=self.re(self.batchnorm2(self.bimap2(x_spd)))
          #x_spd=self.dropout2(x_spd)
          x_spd=self.batchnorm3(self.bimap3(x_spd))
          feature = x_spd
          x_vec=self.logeig(x_spd).view(x_spd.shape[0],-1)
          x_vec=self.dropout3(x_vec)
          y=self.linear(x_vec.float())
          return y, feature

