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
ratio_list = ['0.01-1.00', '0.10-1.00', '1.00-1.00', '10.00-1.00', '100.00-1.00']
# ratio_list = ['0.01-1.00', '0.10-1.00', '1.00-1.00', '100.00-1.00']
# ratio_list = ['0.01-1.00', '0.10-1.00', '1.00-1.00', '10.00-1.00', '100.00-1.00']
# ratio_list = ['10.00-1.00']
# 1/9 轮数低，加轮数到50试试
data_type = '_only_m_a'  # _e_a_density, _a_a_density_ratio  _only_b  _a_a_density_AE _all
frame_len = 28

# x_train = np.load('data/X_train.npy', allow_pickle=True)
# test_normal = np.load('data/X_val_normal.npy', allow_pickle=True)
# test_abnormal = np.load('data/X_val_abnormal.npy', allow_pickle=True)
#
# print(x_train.shape, test_abnormal.shape, test_normal.shape)


def Loss_old_AUC(ratio):
    auc_ae_loss = []
    auc_other_loss = []
    for i in range(EPOCH_MAX):
        all_val1_loss = np.load('clustering/epoch/' + str(i) + '/all_val1_loss' + ratio + '.npy', allow_pickle=True)
        all_val2_loss = np.load('clustering/epoch/' + str(i) + '/all_val2_loss' + ratio + '.npy', allow_pickle=True)
        all_test1_loss = np.load('clustering/epoch/' + str(i) + '/all_test1_loss' + ratio + '.npy', allow_pickle=True)
        all_test2_loss = np.load('clustering/epoch/' + str(i) + '/all_test2_loss' + ratio + '.npy', allow_pickle=True)

        y_true_ae = [0] * all_val1_loss.shape[0] + [1] * all_test1_loss.shape[0]
        y_pred_ae = np.concatenate((all_val1_loss, all_test1_loss), axis=0)
        y_pred_ae = np.array(y_pred_ae)
        y_true_other = [0] * all_val2_loss.shape[0] + [1] * all_test2_loss.shape[0]
        y_pred_other = np.concatenate((all_val2_loss, all_test2_loss), axis=0)
        y_pred_other = np.array(y_pred_other)

        auc_ae_ave = roc_auc_score(y_true_ae, y_pred_ae)
        auc_other_ave = roc_auc_score(y_true_other, y_pred_other)
        auc_ae_loss.append(auc_ae_ave)
        auc_other_loss.append(auc_other_ave)

    return auc_ae_loss, auc_other_loss


def Loss_AUC(ratio):
    auc_ae_loss_ave = []
    auc_other_loss_ave = []
    auc_ae_loss_max = []
    auc_other_loss_max = []
    auc_ae_loss_amax = []
    auc_other_loss_amax = []
    auc_ae_loss_min = []
    auc_other_loss_min = []
    auc_ae_loss_amin = []
    auc_other_loss_amin = []

    pauc_ae_loss_ave = []
    pauc_other_loss_ave = []
    pauc_ae_loss_max = []
    pauc_other_loss_max = []
    pauc_ae_loss_amax = []
    pauc_other_loss_amax = []
    pauc_ae_loss_min = []
    pauc_other_loss_min = []
    pauc_ae_loss_amin = []
    pauc_other_loss_amin = []

    for i in range(EPOCH_MAX):
        # AE
        # all_val1_loss = np.load('clustering_ae/AE/epoch/' + str(i) + '/all_val1_loss' + ratio + '.npy', allow_pickle=True)
        # all_val2_loss = np.load('clustering_ae/AE/epoch/' + str(i) + '/all_val2_loss' + ratio + '.npy', allow_pickle=True)
        # all_test1_loss = np.load('clustering_ae/AE/epoch/' + str(i) + '/all_test1_loss' + ratio + '.npy', allow_pickle=True)
        # all_test2_loss = np.load('clustering_ae/AE/epoch/' + str(i) + '/all_test2_loss' + ratio + '.npy', allow_pickle=True)
        # vAE
        all_val1_loss = np.load('clustering' + data_type + '/''/epoch/' + str(i) + '/all_val1_loss' + ratio + '.npy', allow_pickle=True)
        all_val2_loss = np.load('clustering' + data_type + '/''/epoch/' + str(i) + '/all_val2_loss' + ratio + '.npy', allow_pickle=True)
        all_test1_loss = np.load('clustering' + data_type + '/''/epoch/' + str(i) + '/all_test1_loss' + ratio + '.npy', allow_pickle=True)
        all_test2_loss = np.load('clustering' + data_type + '/''/epoch/' + str(i) + '/all_test2_loss' + ratio + '.npy', allow_pickle=True)
        # ############## average ################
        all_val1_ave = all_val1_loss
        all_val2_ave = all_val2_loss
        all_test1_ave = all_test1_loss
        all_test2_ave = all_test2_loss

        # all_val1_ave = np.array([sum(x) / len(x) for x in chunked(all_val1_ave, 28)])
        # all_val2_ave = np.array([sum(x) / len(x) for x in chunked(all_val2_ave, 28)])
        # all_test1_ave = np.array([sum(x) / len(x) for x in chunked(all_test1_ave, 28)])
        # all_test2_ave = np.array([sum(x) / len(x) for x in chunked(all_test2_ave, 28)])
        all_val1_ave = np.array([sum(x) for x in chunked(all_val1_ave, frame_len)])
        all_val2_ave = np.array([sum(x) for x in chunked(all_val2_ave, frame_len)])
        all_test1_ave = np.array([sum(x) for x in chunked(all_test1_ave, frame_len)])
        all_test2_ave = np.array([sum(x) for x in chunked(all_test2_ave, frame_len)])
        # auc
        y_true_ae_ave = [0] * all_val1_ave.shape[0] + [1] * all_test1_ave.shape[0]
        y_pred_ae_ave = np.concatenate((all_val1_ave, all_test1_ave), axis=0)
        y_pred_ae_ave = np.array(y_pred_ae_ave)
        y_true_other_ave = [0] * all_val2_ave.shape[0] + [1] * all_test2_ave.shape[0]
        y_pred_other_ave = np.concatenate((all_val2_ave, all_test2_ave), axis=0)
        y_pred_other_ave = np.array(y_pred_other_ave)
        # auc
        auc_ae_ave = roc_auc_score(y_true_ae_ave, y_pred_ae_ave)
        auc_other_ave = roc_auc_score(y_true_other_ave, y_pred_other_ave)
        auc_ae_loss_ave.append(auc_ae_ave)
        auc_other_loss_ave.append(auc_other_ave)
        # auc画图
        fpr, tpr, _ = roc_curve(y_true_ae_ave, y_pred_ae_ave)

        # pauc
        pauc_ae_ave = roc_auc_score(y_true_ae_ave, y_pred_ae_ave, max_fpr=0.1)
        pauc_other_ave = roc_auc_score(y_true_other_ave, y_pred_other_ave, max_fpr=0.1)
        pauc_ae_loss_ave.append(pauc_ae_ave)
        pauc_other_loss_ave.append(pauc_other_ave)
        # ############## min ####################
        all_val1_min = all_val1_loss
        all_val2_min = all_val2_loss
        all_test1_min = all_test1_loss
        all_test2_min = all_test2_loss

        # all_val1_max = np.array([max(x) for x in chunked(all_val1_max, 28)])
        # all_val2_max = np.array([max(x) for x in chunked(all_val2_max, 28)])
        # all_test1_max = np.array([max(x) for x in chunked(all_test1_max, 28)])
        # all_test2_max = np.array([max(x) for x in chunked(all_test2_max, 28)])
        all_val1_min = np.array([min(x) for x in chunked(all_val1_min, frame_len)])
        all_val2_min = np.array([min(x) for x in chunked(all_val2_min, frame_len)])
        all_test1_min = np.array([min(x) for x in chunked(all_test1_min, frame_len)])
        all_test2_min = np.array([min(x) for x in chunked(all_test2_min, frame_len)])

        y_true_ae_min = [0] * all_val1_min.shape[0] + [1] * all_test1_min.shape[0]
        y_pred_ae_min = np.concatenate((all_val1_min, all_test1_min), axis=0)
        y_pred_ae_min = np.array(y_pred_ae_min)
        y_true_other_min = [0] * all_val2_min.shape[0] + [1] * all_test2_min.shape[0]
        y_pred_other_min = np.concatenate((all_val2_min, all_test2_min), axis=0)
        y_pred_other_min = np.array(y_pred_other_min)
        # auc
        auc_ae_min = roc_auc_score(y_true_ae_min, y_pred_ae_min)
        auc_other_min = roc_auc_score(y_true_other_min, y_pred_other_min)
        auc_ae_loss_min.append(auc_ae_min)
        auc_other_loss_min.append(auc_other_min)
        # pauc
        pauc_ae_min = roc_auc_score(y_true_ae_min, y_pred_ae_min, max_fpr=0.1)
        pauc_other_min = roc_auc_score(y_true_other_min, y_pred_other_min, max_fpr=0.1)
        pauc_ae_loss_min.append(pauc_ae_min)
        pauc_other_loss_min.append(pauc_other_min)
        # ############## the average of t min ###################
        t = 5
        all_val1_amin = all_val1_loss
        all_val2_amin = all_val2_loss
        all_test1_amin = all_test1_loss
        all_test2_amin = all_test2_loss

        all_val1_amin = [sorted(x, reverse=False) for x in chunked(all_val1_amin, frame_len)]
        all_val2_amin = [sorted(x, reverse=False) for x in chunked(all_val2_amin, frame_len)]
        all_test1_amin = [sorted(x, reverse=False) for x in chunked(all_test1_amin, frame_len)]
        all_test2_amin = [sorted(x, reverse=False) for x in chunked(all_test2_amin, frame_len)]
        all_val1_amin = np.array([sum(x[:t]) for x in all_val1_amin])
        all_val2_amin = np.array([sum(x[:t]) for x in all_val2_amin])
        all_test1_amin = np.array([sum(x[:t]) for x in all_test1_amin])
        all_test2_amin = np.array([sum(x[:t]) for x in all_test2_amin])

        # k1 = 10
        # k2 = 16
        # all_val1_amin = np.array([sum(x[k1:k2]) for x in all_val1_amin])
        # all_val2_amin = np.array([sum(x[k1:k2]) for x in all_val2_amin])
        # all_test1_amin = np.array([sum(x[k1:k2]) for x in all_test1_amin])
        # all_test2_amin = np.array([sum(x[k1:k2]) for x in all_test2_amin])

        y_true_ae_amin = [0] * all_val1_amin.shape[0] + [1] * all_test1_amin.shape[0]
        y_pred_ae_amin = np.concatenate((all_val1_amin, all_test1_amin), axis=0)
        y_pred_ae_amin = np.array(y_pred_ae_amin)
        y_true_other_amin = [0] * all_val2_amin.shape[0] + [1] * all_test2_amin.shape[0]
        y_pred_other_amin = np.concatenate((all_val2_amin, all_test2_amin), axis=0)
        y_pred_other_amin = np.array(y_pred_other_amin)

        auc_ae_amin = roc_auc_score(y_true_ae_amin, y_pred_ae_amin)
        auc_other_amin = roc_auc_score(y_true_other_amin, y_pred_other_amin)
        auc_ae_loss_amin.append(auc_ae_amin)
        auc_other_loss_amin.append(auc_other_amin)

        pauc_ae_amin = roc_auc_score(y_true_ae_amin, y_pred_ae_amin, max_fpr=0.1)
        pauc_other_amin = roc_auc_score(y_true_other_amin, y_pred_other_amin, max_fpr=0.1)
        pauc_ae_loss_amin.append(pauc_ae_amin)
        pauc_other_loss_amin.append(pauc_other_amin)
        # ############## max ####################
        all_val1_max = all_val1_loss
        all_val2_max = all_val2_loss
        all_test1_max = all_test1_loss
        all_test2_max = all_test2_loss

        all_val1_max = np.array([max(x) for x in chunked(all_val1_max, frame_len)])
        all_val2_max = np.array([max(x) for x in chunked(all_val2_max, frame_len)])
        all_test1_max = np.array([max(x) for x in chunked(all_test1_max, frame_len)])
        all_test2_max = np.array([max(x) for x in chunked(all_test2_max, frame_len)])

        y_true_ae_max = [0] * all_val1_max.shape[0] + [1] * all_test1_max.shape[0]
        y_pred_ae_max = np.concatenate((all_val1_max, all_test1_max), axis=0)
        y_pred_ae_max = np.array(y_pred_ae_max)
        y_true_other_max = [0] * all_val2_max.shape[0] + [1] * all_test2_max.shape[0]
        y_pred_other_max = np.concatenate((all_val2_max, all_test2_max), axis=0)
        y_pred_other_max = np.array(y_pred_other_max)

        auc_ae_max = roc_auc_score(y_true_ae_max, y_pred_ae_max)
        auc_other_max = roc_auc_score(y_true_other_max, y_pred_other_max)
        auc_ae_loss_max.append(auc_ae_max)
        auc_other_loss_max.append(auc_other_max)

        pauc_ae_max = roc_auc_score(y_true_ae_max, y_pred_ae_max, max_fpr=0.1)
        pauc_other_max = roc_auc_score(y_true_other_max, y_pred_other_max, max_fpr=0.1)
        pauc_ae_loss_max.append(pauc_ae_max)
        pauc_other_loss_max.append(pauc_other_max)
        # ############## the average of t max ###################
        all_val1_amax = all_val1_loss
        all_val2_amax = all_val2_loss
        all_test1_amax = all_test1_loss
        all_test2_amax = all_test2_loss

        all_val1_amax = [sorted(x, reverse=True) for x in chunked(all_val1_amax, frame_len)]
        all_val2_amax = [sorted(x, reverse=True) for x in chunked(all_val2_amax, frame_len)]
        all_test1_amax = [sorted(x, reverse=True) for x in chunked(all_test1_amax, frame_len)]
        all_test2_amax = [sorted(x, reverse=True) for x in chunked(all_test2_amax, frame_len)]
        all_val1_amax = np.array([sum(x[:t]) for x in all_val1_amax])
        all_val2_amax = np.array([sum(x[:t]) for x in all_val2_amax])
        all_test1_amax = np.array([sum(x[:t]) for x in all_test1_amax])
        all_test2_amax = np.array([sum(x[:t]) for x in all_test2_amax])

        y_true_ae_amax = [0] * all_val1_amax.shape[0] + [1] * all_test1_amax.shape[0]
        y_pred_ae_amax = np.concatenate((all_val1_amax, all_test1_amax), axis=0)
        y_pred_ae_amax = np.array(y_pred_ae_amax)
        y_true_other_amax = [0] * all_val2_amax.shape[0] + [1] * all_test2_amax.shape[0]
        y_pred_other_amax = np.concatenate((all_val2_amax, all_test2_amax), axis=0)
        y_pred_other_amax = np.array(y_pred_other_amax)

        auc_ae_amax = roc_auc_score(y_true_ae_amax, y_pred_ae_amax)
        auc_other_amax = roc_auc_score(y_true_other_amax, y_pred_other_amax)
        auc_ae_loss_amax.append(auc_ae_amax)
        auc_other_loss_amax.append(auc_other_amax)

        pauc_ae_amax = roc_auc_score(y_true_ae_amax, y_pred_ae_amax, max_fpr=0.1)
        pauc_other_amax = roc_auc_score(y_true_other_amax, y_pred_other_amax, max_fpr=0.1)
        pauc_ae_loss_amax.append(pauc_ae_amax)
        pauc_other_loss_amax.append(pauc_other_amax)

    return auc_ae_loss_ave, auc_other_loss_ave, pauc_ae_loss_ave, pauc_other_loss_ave, \
           auc_ae_loss_min, auc_other_loss_min, pauc_ae_loss_min, pauc_other_loss_min, \
           auc_ae_loss_amin, auc_other_loss_amin, pauc_ae_loss_amin, pauc_other_loss_amin, \
           auc_ae_loss_max, auc_other_loss_max, pauc_ae_loss_max, pauc_other_loss_max, \
           auc_ae_loss_amax, auc_other_loss_amax, pauc_ae_loss_amax, pauc_other_loss_amax


def Latent_AUC(ratio):
    all_auc = []
    all_other_auc = []
    pall_auc = []
    pall_other_auc = []
    gmm = mixture.GaussianMixture()
    for i in range(EPOCH_MAX):
        training_latent = np.load('clustering' + data_type + '/''/epoch/' + str(i) + '/all_training_latent' + ratio + '.npy', allow_pickle=True)
        test_normal_latent = np.load('clustering' + data_type + '/''/epoch/' + str(i) + '/all_val1_latent' + ratio + '.npy', allow_pickle=True)
        test_abnormal_latent = np.load('clustering' + data_type + '/''/epoch/' + str(i) + '/all_test1_latent' + ratio + '.npy', allow_pickle=True)
        test_other_normal_latent = np.load('clustering' + data_type + '/''/epoch/' + str(i) + '/all_val2_latent' + ratio + '.npy', allow_pickle=True)
        test_other_abnormal_latent = np.load('clustering' + data_type + '/''/epoch/' + str(i) + '/all_test2_latent' + ratio + '.npy', allow_pickle=True)
        # training_latent = training_latent.reshape(-1, 30)
        test_normal_latent = test_normal_latent.reshape(-1, 16)  # 16为隐藏层size
        test_abnormal_latent = test_abnormal_latent.reshape(-1, 16)
        test_other_normal_latent = test_other_normal_latent.reshape(-1, 16)  # 16为隐藏层size
        test_other_abnormal_latent = test_other_abnormal_latent.reshape(-1, 16)
        gmm.fit(training_latent)
        llh1 = gmm.score_samples(training_latent)  # 输入(n_samples, n_features), 输出(n_samples,)
        llh2 = gmm.score_samples(test_normal_latent)
        llh3 = gmm.score_samples(test_abnormal_latent)
        llh4 = gmm.score_samples(test_other_normal_latent)
        llh5 = gmm.score_samples(test_other_abnormal_latent)

        llh1 = llh1.reshape(-1, batch_size)
        llh2 = llh2.reshape(-1, batch_size)
        llh3 = llh3.reshape(-1, batch_size)
        llh4 = llh4.reshape(-1, batch_size)
        llh5 = llh5.reshape(-1, batch_size)
        llh1_llh2 = np.mean(llh1) - np.mean(llh2, axis=1)  # 求每一行均值
        llh1_llh3 = np.mean(llh1) - np.mean(llh3, axis=1)  # 求每一行均值
        llh1_llh4 = np.mean(llh1) - np.mean(llh4, axis=1)  # 求每一行均值
        llh1_llh5 = np.mean(llh1) - np.mean(llh5, axis=1)  # 求每一行均值

        y_true = [0]*(test_normal_latent.shape[0]//batch_size) + [1]*(test_abnormal_latent.shape[0]//batch_size)
        y_pred = np.concatenate((llh1_llh2, llh1_llh3), axis=0)
        y_pred = np.array(y_pred)
        auc = roc_auc_score(y_true, y_pred)
        pauc = roc_auc_score(y_true, y_pred, max_fpr=0.1)
        all_auc.append(auc)
        pall_auc.append(pauc)

        y_true_other = [0]*(test_other_normal_latent.shape[0]//batch_size) + [1]*(test_other_abnormal_latent.shape[0]//batch_size)
        y_pred_other = np.concatenate((llh1_llh4, llh1_llh5), axis=0)
        y_pred_other = np.array(y_pred_other)
        auc_other = roc_auc_score(y_true_other, y_pred_other)
        pauc_other = roc_auc_score(y_true_other, y_pred_other, max_fpr=0.1)
        all_other_auc.append(auc_other)
        pall_other_auc.append(pauc_other)

    return all_auc, pall_auc, all_other_auc, pall_other_auc


def kl_AUC(ratio):
    all_auc = []
    all_other_auc = []
    pall_auc = []
    pall_other_auc = []
    for i in range(EPOCH_MAX):
        test_normal_latent = np.load('clustering' + data_type + '/''/epoch/' + str(i) + '/all_val1_kl_loss' + ratio + '.npy', allow_pickle=True)
        test_abnormal_latent = np.load('clustering' + data_type + '/''/epoch/' + str(i) + '/all_test1_kl_loss' + ratio + '.npy', allow_pickle=True)

        y_true = [0]*(test_normal_latent.shape[0]) + [1]*(test_abnormal_latent.shape[0])
        y_pred = np.concatenate((test_normal_latent, test_abnormal_latent), axis=0)
        y_pred = np.array(y_pred)
        auc = roc_auc_score(y_true, y_pred)
        all_auc.append(auc)
        pauc = roc_auc_score(y_true, y_pred, max_fpr=0.1)
        pall_auc.append(pauc)

        test_other_normal_latent = np.load('clustering' + data_type + '/''/epoch/' + str(i) + '/all_val2_kl_loss' + ratio + '.npy', allow_pickle=True)
        test_other_abnormal_latent = np.load('clustering' + data_type + '/''/epoch/' + str(i) + '/all_test2_kl_loss' + ratio + '.npy', allow_pickle=True)

        y_true_other = [0]*(test_other_normal_latent.shape[0]) + [1]*(test_other_abnormal_latent.shape[0])
        y_pred_other = np.concatenate((test_other_normal_latent, test_other_abnormal_latent), axis=0)
        y_pred_other = np.array(y_pred_other)
        auc_other = roc_auc_score(y_true_other, y_pred_other)
        all_other_auc.append(auc_other)
        pauc_other = roc_auc_score(y_true_other, y_pred_other, max_fpr=0.1)
        pall_other_auc.append(pauc_other)

    return all_auc, pall_auc, all_other_auc, pall_other_auc


def latent_Distance():
    auc_distance = []

    for epoch in range(EPOCH_MAX):
        latent_normal_distance = []
        latent_abnormal_distance = []

        training_latent = np.load('clustering/epoch/' + str(epoch) + '/all_training_latent.npy', allow_pickle=True)
        test_normal_latent = np.load('clustering/epoch/' + str(epoch) + '/all_val_latent.npy', allow_pickle=True)
        test_abnormal_latent = np.load('clustering/epoch/' + str(epoch) + '/all_test_latent.npy', allow_pickle=True)
        training_latent = training_latent.reshape(-1, 28, 30)
        training_latent = np.mean(training_latent, axis=0)

        for i in range(test_normal_latent.shape[0]):
            distance_normal = cdist(training_latent, test_normal_latent[i], metric='euclidean')
            latent_normal_distance.append(np.mean(distance_normal))
        for i in range(test_abnormal_latent.shape[0]):
            distance_abnormal = cdist(training_latent, test_abnormal_latent[i], metric='euclidean')
            latent_abnormal_distance.append(np.mean(distance_abnormal))

        y_true_distance = [0] * test_normal_latent.shape[0] + [1] * test_abnormal_latent.shape[0]
        y_pred_distance = np.concatenate((np.array(latent_normal_distance), np.array(latent_abnormal_distance)), axis=0)
        y_pred_distance = np.array(y_pred_distance)
        auc_distance.append(roc_auc_score(y_true_distance, y_pred_distance))
    return auc_distance


def plot_loss(ratio, train_loss, val1_loss, test1_loss, val2_loss,  test2_loss,
              auc_ae_ave, auc_other_ave, auc_ae_amin, auc_other_amin, auc_ae_amax, auc_other_amax):

    fig = plt.figure()
    font0 = {'family': 'Times New Roman', 'weight': 'normal', 'style': 'italic', 'size': 18}
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}
    font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
    font3 = {'family': 'Times New Roman', 'weight': 'normal', 'style': 'italic', 'size': 15}
    ax1 = fig.add_subplot()
    ax2 = ax1.twinx()
    lns1 = ax1.plot(train_loss,  label='train_loss', color='r')
    lns2 = ax1.plot(val1_loss,  label='test_e_normal', color='b')
    lns3 = ax1.plot(test1_loss, label='test_e_abnormal', color='y')
    lns4 = ax1.plot(val2_loss,  label='test_other_normal', color='purple')
    lns5 = ax1.plot(test2_loss, label='test_other_abnormal', color='pink')

    lns6 = ax2.plot(auc_ae_ave, label='AUC_e_ave', color='g')
    lns7 = ax2.plot(auc_other_ave, label='AUC_other_ave', color='cyan')
    lns8 = ax2.plot(auc_ae_amin, label='AUC_e_amin', color='k')
    lns9 = ax2.plot(auc_other_amin, label='AUC_other_amin', color='tan')
    lns10 = ax2.plot(auc_ae_amax, label='AUC_e_amax', color='lime')
    lns11 = ax2.plot(auc_other_amax, label='AUC_other_max', color='coral')

    plt.title('learning_rate_0.0001', font3)
    lns = lns1 + lns2 + lns3 + lns4 + lns5 + lns6 + lns7 + lns8 + lns9 + lns10 + lns11
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    ax1.set_xlabel('Epoch', font3)
    ax1.set_ylabel('Loss', font3)
    ax2.set_ylabel('AUC', font3)
    ax2.set_ylim(0, 1)
    plt.savefig('clustering_only_e/loss_learning_rate_0.0001_' + ratio + '.png')
    plt.show()

    fig = plt.figure()
    font0 = {'family': 'Times New Roman', 'weight': 'normal', 'style': 'italic', 'size': 18}
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}
    font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
    font3 = {'family': 'Times New Roman', 'weight': 'normal', 'style': 'italic', 'size': 15}
    ax1 = fig.add_subplot()
    ax2 = ax1.twinx()
    lns1 = ax1.plot(np.log2(train_loss),  label='train_loss', color='r')
    lns2 = ax1.plot(np.log2(val1_loss),  label='test_e_normal', color='b')
    lns3 = ax1.plot(np.log2(test1_loss), label='test_e_abnormal', color='y')
    lns4 = ax1.plot(np.log2(val2_loss),  label='test_other_normal', color='purple')
    lns5 = ax1.plot(np.log2(test2_loss), label='test_other_abnormal', color='pink')
    lns6 = ax2.plot(auc_ae_ave, label='AUC_e_ave', color='g')
    lns7 = ax2.plot(auc_other_ave, label='AUC_other_ave', color='cyan')
    lns8 = ax2.plot(auc_ae_amin, label='AUC_e_amin', color='k')
    lns9 = ax2.plot(auc_other_amin, label='AUC_other_amin', color='tan')
    lns10 = ax2.plot(auc_ae_amax, label='AUC_e_amax', color='lime')
    lns11 = ax2.plot(auc_other_amax, label='AUC_other_max', color='coral')
    plt.title('learning_rate_0.0001', font3)
    lns = lns1 + lns2 + lns3 + lns4 + lns5 + lns6 + lns7 + lns8 + lns9 + lns10 + lns11
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    ax1.set_xlabel('Epoch', font3)
    ax1.set_ylabel('Loss', font3)
    ax2.set_ylabel('AUC', font3)
    ax2.set_ylim(0, 1)
    plt.savefig('clustering_only_e/log2_loss_learning_rate_0.0001_' + ratio + '.png')
    plt.show()


def plot_kl_loss(ratio, train_kl_loss, val1_kl_loss, test1_kl_loss, val2_kl_loss,  test2_kl_loss,
                 auc_ae_ave, auc_other_ave, auc_ae_amin, auc_other_amin, auc_ae_amax, auc_other_amax):

    fig = plt.figure()
    font0 = {'family': 'Times New Roman', 'weight': 'normal', 'style': 'italic', 'size': 18}
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}
    font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
    font3 = {'family': 'Times New Roman', 'weight': 'normal', 'style': 'italic', 'size': 15}
    ax1 = fig.add_subplot()
    ax2 = ax1.twinx()
    lns1 = ax1.plot(train_kl_loss,  label='train_kl_loss', color='r')
    lns2 = ax1.plot(val1_kl_loss,  label='test_e_kl_normal', color='b')
    lns3 = ax1.plot(test1_kl_loss, label='test_e_kl_abnormal', color='y')
    lns4 = ax1.plot(val2_kl_loss,  label='test_other_kl_normal', color='purple')
    lns5 = ax1.plot(test2_kl_loss, label='test_other_kl_abnormal', color='pink')
    lns6 = ax2.plot(auc_ae_ave, label='AUC_e_ave', color='g')
    lns7 = ax2.plot(auc_other_ave, label='AUC_other_ave', color='cyan')
    lns8 = ax2.plot(auc_ae_amin, label='AUC_e_amin', color='k')
    lns9 = ax2.plot(auc_other_amin, label='AUC_other_amin', color='tan')
    lns10 = ax2.plot(auc_ae_amax, label='AUC_e_amax', color='lime')
    lns11 = ax2.plot(auc_other_amax, label='AUC_other_max', color='coral')
    plt.title('learning_rate_0.0001', font3)
    lns = lns1 + lns2 + lns3 + lns4 + lns5 + lns6 + lns7 + lns8 + lns9 + lns10 + lns11
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    ax1.set_xlabel('Epoch', font3)
    ax1.set_ylabel('KL_Loss', font3)
    ax2.set_ylabel('AUC', font3)
    ax2.set_ylim(0, 1)
    plt.savefig('clustering_only_e/kl_loss_learning_rate_0.0001_' + ratio + '.png')
    plt.show()

    fig = plt.figure()
    font0 = {'family': 'Times New Roman', 'weight': 'normal', 'style': 'italic', 'size': 18}
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}
    font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
    font3 = {'family': 'Times New Roman', 'weight': 'normal', 'style': 'italic', 'size': 15}
    ax1 = fig.add_subplot()
    ax2 = ax1.twinx()
    lns1 = ax1.plot(np.log2(train_kl_loss),  label='train_kl_loss', color='r')
    lns2 = ax1.plot(np.log2(val1_kl_loss),  label='test_e_kl_normal', color='b')
    lns3 = ax1.plot(np.log2(test1_kl_loss), label='test_e_kl_abnormal', color='y')
    lns4 = ax1.plot(np.log2(val2_kl_loss),  label='test_other_kl_normal', color='purple')
    lns5 = ax1.plot(np.log2(test2_kl_loss), label='test_other_kl_abnormal', color='pink')
    lns6 = ax2.plot(auc_ae_ave, label='AUC_e_ave', color='g')
    lns7 = ax2.plot(auc_other_ave, label='AUC_other_ave', color='cyan')
    lns8 = ax2.plot(auc_ae_amin, label='AUC_e_amin', color='k')
    lns9 = ax2.plot(auc_other_amin, label='AUC_other_amin', color='tan')
    lns10 = ax2.plot(auc_ae_amax, label='AUC_e_amax', color='lime')
    lns11 = ax2.plot(auc_other_amax, label='AUC_other_max', color='coral')
    plt.title('learning_rate_0.0001', font3)
    lns = lns1 + lns2 + lns3 + lns4 + lns5 + lns6 + lns7 + lns8 + lns9 + lns10 + lns11
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    ax1.set_xlabel('Epoch', font3)
    ax1.set_ylabel('KL_Loss', font3)
    ax2.set_ylabel('AUC', font3)
    ax2.set_ylim(0, 1)
    plt.savefig('clustering_only_e/log2_kl_loss_learning_rate_0.0001_' + ratio + '.png')
    plt.show()


# def Plot_AUC(Loss_AUC, Latent_AUC):
#     fig = plt.figure()
#     ax1 = fig.add_subplot()
#     ax2 = ax1.twinx()
#     lns1 = ax1.plot(Latent_AUC,  label='Latent_AUC', color='r')
#     lns2 = ax2.plot(Loss_AUC, label='Loss_AUC', color='g')
#     ax1.set_xlabel('Epoch 0-99')
#     ax1.set_ylabel('AUC')
#     lns = lns1 + lns2
#     labs = [l.get_label() for l in lns]
#     ax1.legend(lns, labs, loc=0)
#     plt.show()


def Plot_AUC(Loss_AUC, Latent_AUC, auc_distance):
    plt.plot(Loss_AUC, label='Loss_AUC', color='r')
    plt.plot(Latent_AUC, label='Latent_AUC', color='g')
    plt.plot(auc_distance, label='Latent_distance_AUC', color='b')
    plt.legend()
    plt.show()


def print_test_loss():
    n = 5
    for i in range(EPOCH_MAX):
        a = np.load('clustering/epoch/' + str(i) + '/all_test_loss.npy')
        print(a)
        max_indexs = heapq.nlargest(n, range(len(a)), a.take)
        print(max_indexs)


def main():
    print(data_type)
    print('\nloss')
    for ratio in ratio_list:
        print('***********************')
        print(ratio)
        auc_ae_ave, auc_other_ave, pauc_ae_ave, pauc_other_ave, \
        auc_ae_min, auc_other_min, pauc_ae_min, pauc_other_min, \
        auc_ae_amin, auc_other_amin, pauc_ae_amin, pauc_other_amin, \
        auc_ae_max, auc_other_max, pauc_ae_max, pauc_other_max, \
        auc_ae_amax, auc_other_amax, pauc_ae_amax, pauc_other_amax = Loss_AUC(ratio)
        print('********* ave **********')
        # print(auc_ae_ave)
        # print(pauc_ae_ave)
        # print(auc_other_ave)
        # print(pauc_other_ave)
        ave = np.argmax(auc_ae_ave)  # 最大值索引
        # density_loss = np.load('data' + data_type + '/train_density_loss' + ratio + '.npy', allow_pickle=True)
        print('auc_e: {:.4f}  epoch: {:0}'.format(max(auc_ae_ave), np.argmax(auc_ae_ave)))
        # print('density_loss: {:.4f}'.format(density_loss[ave.real]))
        print('pauc_e: {:.4f}'.format(pauc_ae_ave[ave.real]))
        print('auc_other: {:.4f}'.format(auc_other_ave[ave.real]))
        print('pauc_other: {:.4f}'.format(pauc_other_ave[ave.real]))
        print('********* best **********')
        print('auc_e: {:.4f}  epoch: {:0}'.format(max(auc_ae_ave), np.argmax(auc_ae_ave)))
        print('pauc_e: {:.4f}  epoch: {:0}'.format(max(pauc_ae_ave), np.argmax(pauc_ae_ave)))
        print('auc_other: {:.4f}  epoch: {:0}'.format(max(auc_other_ave), np.argmax(auc_other_ave)))
        print('pauc_other: {:.4f}  epoch: {:0}'.format(max(pauc_other_ave), np.argmax(pauc_other_ave)))
    #     # print('********* max **********')
    #     # print(auc_ae_max)
    #     # print(pauc_ae_max)
    #     # print(auc_other_max)
    #     # print(pauc_other_max)
    #     # print('auc_e:', (max(auc_ae_max)), 'epoch:', np.argmax(auc_ae_max))
    #     # print('pauc_e:', (max(pauc_ae_max)), 'epoch:', np.argmax(pauc_ae_max))
    #     # print('auc_other:', (max(auc_other_max)), 'epoch:', np.argmax(auc_other_max))
    #     # print('pauc_other:', (max(pauc_other_max)), 'epoch:', np.argmax(pauc_other_max))
    #     print('********* amax **********')
    #     # print(auc_ae_amax)
    #     # print(pauc_ae_amax)
    #     # print(auc_other_amax)
    #     # print(pauc_other_amax)
    #     amax = np.argmax(auc_ae_amax)
    #     print('auc_e: {:.4f}  epoch: {:0}'.format(max(auc_ae_amax), np.argmax(auc_ae_amax)))
    #     print('pauc_e: {:.4f}'.format(pauc_ae_amax[amax.real]))
    #     print('auc_other: {:.4f}'.format(auc_other_amax[amax.real]))
    #     print('pauc_other: {:.4f}'.format(pauc_other_amax[amax.real]))
    #     print('********* best **********')
    #     print('auc_e: {:.4f}  epoch: {:0}'.format(max(auc_ae_amax), np.argmax(auc_ae_amax)))
    #     print('pauc_e: {:.4f}  epoch: {:0}'.format(max(pauc_ae_amax), np.argmax(pauc_ae_amax)))
    #     print('auc_other: {:.4f}  epoch: {:0}'.format(max(auc_other_amax), np.argmax(auc_other_amax)))
    #     print('pauc_other: {:.4f}  epoch: {:0}'.format(max(pauc_other_amax), np.argmax(pauc_other_amax)))
    #     # print('********* min **********')
    #     # print(auc_ae_min)
    #     # print(pauc_ae_min)
    #     # print(auc_other_min)
    #     # print(pauc_other_min)
    #     # print('auc_e:', (max(auc_ae_min)), 'epoch:', np.argmax(auc_ae_min))
    #     # print('pauc_e:', (max(pauc_ae_min)), 'epoch:', np.argmax(pauc_ae_min))
    #     # print('auc_other:', (max(auc_other_min)), 'epoch:', np.argmax(auc_other_min))
    #     # print('pauc_other:', (max(pauc_other_min)), 'epoch:', np.argmax(pauc_other_min))
    #     print('********* amin **********')
    #     # print(auc_ae_amin)
    #     # print(pauc_ae_amin)
    #     # print(auc_other_amin)
    #     # print(pauc_other_amin)
    #     www = np.argmax(auc_ae_amin)
    #     print('auc_e: {:.4f}  epoch: {:0}'.format(max(auc_ae_amin), np.argmax(auc_ae_amin)))
    #     print('pauc_e: {:.4f}'.format(pauc_ae_amin[www.real]))
    #     print('auc_other: {:.4f}'.format(auc_other_amin[www.real]))
    #     print('pauc_other: {:.4f}'.format(pauc_other_amin[www.real]))
    #     print('********* best **********')
    #     print('auc_e: {:.4f}  epoch: {:0}'.format(max(auc_ae_amin), np.argmax(auc_ae_amin)))
    #     print('pauc_e: {:.4f}  epoch: {:0}'.format(max(pauc_ae_amin), np.argmax(pauc_ae_amin)))
    #     print('auc_other: {:.4f}  epoch: {:0}'.format(max(auc_other_amin), np.argmax(auc_other_amin)))
    #     print('pauc_other: {:.4f}  epoch: {:0}'.format(max(pauc_other_amin), np.argmax(pauc_other_amin)))

    # print('\nlatent')
    # for ratio in ratio_list:
    #     print('***********************')
    #     print(ratio)
    #     auc, pauc, auc_other, pauc_other = Latent_AUC(ratio)
    #     www = np.argmax(auc)
    #     print('auc_e: {:.4f}  epoch: {:0}'.format(max(auc), np.argmax(auc)))
    #     print('pauc_e: {:.4f}'.format(pauc[www.real]))
    #     print('auc_other: {:.4f}'.format(auc_other[www.real]))
    #     print('pauc_other: {:.4f}'.format(pauc_other[www.real]))
    #     print('********* best **********')
    #     print('auc_e: {:.4f}  epoch: {:0}'.format(max(auc), np.argmax(auc)))
    #     print('pauc_e: {:.4f}  epoch: {:0}'.format(max(pauc), np.argmax(pauc)))
    #     print('auc_other: {:.4f}  epoch: {:0}'.format(max(auc_other), np.argmax(auc_other)))
    #     print('pauc_other: {:.4f}  epoch: {:0}'.format(max(pauc_other), np.argmax(pauc_other)))
    #
    # print('\nkl_loss')
    # for ratio in ratio_list:
    #     if ratio == 'AE':
    #         pass
    #     else:
    #         print('***********************')
    #         print(ratio)
    #         auc, pauc, auc_other, pauc_other = kl_AUC(ratio)
    #         www = np.argmax(auc)
    #         print('auc_e: {:.4f}  epoch: {:0}'.format(max(auc), np.argmax(auc)))
    #         print('pauc_e: {:.4f}'.format(pauc[www.real]))
    #         print('auc_other: {:.4f}'.format(auc_other[www.real]))
    #         print('pauc_other: {:.4f}'.format(pauc_other[www.real]))
    #         print('********* best **********')
    #         print('auc_e: {:.4f}  epoch: {:0}'.format(max(auc), np.argmax(auc)))
    #         print('pauc_e: {:.4f}  epoch: {:0}'.format(max(pauc), np.argmax(pauc)))
    #         print('auc_other: {:.4f}  epoch: {:0}'.format(max(auc_other), np.argmax(auc_other)))
    #         print('pauc_other: {:.4f}  epoch: {:0}'.format(max(pauc_other), np.argmax(pauc_other)))


if __name__ == '__main__':
    main()
    # print_test_loss()

