import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn import mixture
from scipy.spatial.distance import cdist
from scipy.stats import norm
import scipy.io as sio
from utils import *
from matplotlib import pyplot
import matplotlib.pyplot as plt
import math
import pylab as pl
import heapq
from more_itertools import chunked

# Hyper parameter
task = 'PCG (PhysioNet)'
EPOCH_MAX = 50  # 100
dropout = 0
batch_size = 28
input_size = 70  # 14 * 5 frequency domain 5 frames altogether

ratio_list = ['AE', '0.01-1.00', '1.00-1.00']
epoch_list = [13, 13, 7]
# ratio_list = ['AE', '0.01-1.00', '1.00-1.00']
# 1/9 轮数低，加轮数到50试试
# data_type = ['_all', '_ae', '_ef']
data_type = ['_ef']
frame_len = 28
i = 0
fpr = [[], [], []]
tpr = [[], [], []]
for data_t in data_type:

    for ratio in ratio_list:

        all_val1_loss = np.load('clustering' + data_t + '/''/epoch/' + str(epoch_list[i]) + '/all_val1_loss' + ratio + '.npy', allow_pickle=True)
        all_test1_loss = np.load('clustering' + data_t + '/''/epoch/' + str(epoch_list[i]) + '/all_test1_loss' + ratio + '.npy', allow_pickle=True)
        # ############## average ################
        all_val1_ave = all_val1_loss
        all_test1_ave = all_test1_loss

        all_val1_ave = np.array([sum(x) for x in chunked(all_val1_ave, frame_len)])
        all_test1_ave = np.array([sum(x) for x in chunked(all_test1_ave, frame_len)])
        # auc
        y_true_ae_ave = [0] * all_val1_ave.shape[0] + [1] * all_test1_ave.shape[0]
        y_pred_ae_ave = np.concatenate((all_val1_ave, all_test1_ave), axis=0)
        y_pred_ae_ave = np.array(y_pred_ae_ave)

        fpr[i], tpr[i], _ = roc_curve(y_true_ae_ave, y_pred_ae_ave)

        i = i + 1

plt.figure()
lw = 2
font1 = {'family': 'Times New Roman',
'weight' : 'normal',
'size' : 16,
}
plt.plot(fpr[0], tpr[0], lw=lw, label='β=AE', linestyle='--')
plt.plot(fpr[1], tpr[1], lw=lw, label='β=0.01')
plt.plot(fpr[2], tpr[2], lw=lw, label='β=1', linestyle=':')
plt.xlabel('False Positive Rate', font1)
plt.ylabel('True Positive Rate', font1)
plt.tick_params(labelsize=13)
plt.title("AUC of subset 'ef '", font1)
plt.legend(loc="lower right", fontsize='12')
plt.show()
