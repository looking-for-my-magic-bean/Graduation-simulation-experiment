import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
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

kl_ratio = 0.9
loss_ratio = 0.1
# ratio = str('%.2f-%.2f' % (kl_ratio, loss_ratio))
ratio_list = ['0.1-0.9', '0.3-0.7', '0.5-0.5', '0.7-0.3', '0.9-0.1', '0.01-0.99', '0.99-0.01']


def calc_corr(a, b):
    a_avg = sum(a) / len(a)
    b_avg = sum(b) / len(b)

    # 计算分子，协方差————按照协方差公式，本来要除以n的，由于在相关系数中上下同时约去了n，于是可以不除以n
    cov_ab = sum([(x - a_avg) * (y - b_avg) for x, y in zip(a, b)])

    # 计算分母，方差乘积————方差本来也要除以n，在相关系数中上下同时约去了n，于是可以不除以n
    sq = math.sqrt(sum([(x - a_avg) ** 2 for x in a]) * sum([(x - b_avg) ** 2 for x in b]))

    corr_factor = cov_ab / sq

    return corr_factor


for ratio in ratio_list:
    training_loss_plot = np.load('data/train_loss' + ratio + '.npy', allow_pickle=True)
    test_ae_normal_plot = np.load('data/test_ae_normal_loss' + ratio + '.npy', allow_pickle=True)
    test_ae_abnormal_plot = np.load('data/test_ae_abnormal_loss' + ratio + '.npy', allow_pickle=True)
    test_other_normal_plot = np.load('data/test_other_normal_loss' + ratio + '.npy', allow_pickle=True)
    test_other_abnormal_plot = np.load('data/test_other_abnormal_loss' + ratio + '.npy', allow_pickle=True)
    test_ae_loss = np.concatenate((test_ae_normal_plot, test_ae_abnormal_plot), axis=0)
    test_other_loss = np.concatenate((test_ae_normal_plot, test_ae_abnormal_plot), axis=0)

    training_kl_loss_plot = np.load('data/train_kl_loss' + ratio + '.npy', allow_pickle=True)
    test_ae_normal_kl_plot = np.load('data/test_ae_normal_kl_loss' + ratio + '.npy', allow_pickle=True)
    test_ae_abnormal_kl_plot = np.load('data/test_ae_abnormal_kl_loss' + ratio + '.npy', allow_pickle=True)
    test_other_normal_kl_plot = np.load('data/test_other_normal_kl_loss' + ratio + '.npy', allow_pickle=True)
    test_other_abnormal_kl_plot = np.load('data/test_other_abnormal_kl_loss' + ratio + '.npy', allow_pickle=True)
    test_ae_kl = np.concatenate((test_ae_normal_kl_plot, test_ae_abnormal_kl_plot), axis=0)
    test_other_kl = np.concatenate((test_other_normal_kl_plot, test_other_abnormal_kl_plot), axis=0)

    # training_loss_plot = training_loss_plot.tolist()
    training = calc_corr(training_loss_plot, training_kl_loss_plot)
    test_ae_normal = calc_corr(test_ae_normal_plot, test_ae_normal_kl_plot)
    test_ae_abnormal = calc_corr(test_ae_abnormal_plot, test_ae_abnormal_kl_plot)
    test_other_normal = calc_corr(test_other_normal_plot, test_other_normal_kl_plot)
    test_other_abnormal = calc_corr(test_other_abnormal_plot, test_other_abnormal_kl_plot)

    test_ae = calc_corr(test_ae_loss, test_other_loss)
    test_other = calc_corr(test_ae_kl, test_other_kl)

    print('******************* ratio: {0} ********************************\n'
          'Training: {1:.5f}\n'
          'test_ae_normal: {2:.5f}              test_ae_abnormal: {3:.5f}\n'
          'test_other_normal: {4:.5f}           test_other_abnormal: {5:.5f}\n'
          'test_ae: {6:.5f}           test_other: {7:.5f}\n'
          .format(ratio, training, test_ae_normal, test_ae_abnormal, test_other_normal, test_other_abnormal, test_ae, test_other))


