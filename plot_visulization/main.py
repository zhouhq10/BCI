import scipy.io as scio
import numpy as np
from pathlib import Path
import os
import random

from utlis import *
from train import *
import torch as th
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
#plotly.offline.init_notebook_mode()
import matplotlib.pyplot as plt

feature1, feature2 = BciNet(train_loader, test_loader)
logrie1 = extract_feature(feature1)
logrie2 = extract_feature(feature2)

projections1 = TSNE(n_components=2).fit_transform(logrie1)
projections2 = TSNE(n_components=2).fit_transform(logrie2)

plt.scatter(projections1[:,0], projections1[:,1], c='mediumblue',label='session1')
plt.scatter(projections2[:,0], projections2[:,1], c='darkorange',label='session2')