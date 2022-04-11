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
from more_itertools import chunked


# Hyper parameter
task = 'PCG (PhysioNet)'
EPOCH_MAX = 50  # 100
dropout = 0
batch_size = 28
input_size = 70  # 14 * 5 frequency domain 5 frames altogether
# ratio_list = ['1.00-1.00', '0.10-1.00', '0.01-1.00', 'AE', '0.00-1.00']
# ratio_list = ['1.00-1.00', '0.10-1.00', '0.01-1.00', '0.00-1.00']
ratio_list = ['0.01-1.00', '0.10-1.00', '1.00-1.00', '10.00-1.00', '100.00-1.00']
# ratio_list = ['AE', '0.01-1.00', '1.00-1.00']
# 1/9 轮数低，加轮数到50试试
data_type = '_only_e'
frame_len = 28


def Loss_AUC(ratio):
    auc_e_loss_amin = []
    auc_a_loss_amin = []
    auc_b_loss_amin = []
    auc_c_loss_amin = []
    auc_d_loss_amin = []
    auc_e_loss_amin = []
    auc_f_loss_amin = []
    auc_other_loss_amin = []

    pauc_e_loss_amin = []
    pauc_a_loss_amin = []
    pauc_b_loss_amin = []
    pauc_c_loss_amin = []
    pauc_d_loss_amin = []
    pauc_e_loss_amin = []
    pauc_f_loss_amin = []
    pauc_other_loss_amin = []

    for i in range(EPOCH_MAX):
        # vAE
        all_val1_loss = np.load('clustering' + data_type + '/''/epoch/' + str(i) + '/all_val1_loss' + ratio + '.npy', allow_pickle=True)
        all_val2_loss = np.load('clustering' + data_type + '/''/epoch/' + str(i) + '/all_val2_loss' + ratio + '.npy', allow_pickle=True)
        all_test1_loss = np.load('clustering' + data_type + '/''/epoch/' + str(i) + '/all_test1_loss' + ratio + '.npy', allow_pickle=True)
        all_test2_loss = np.load('clustering' + data_type + '/''/epoch/' + str(i) + '/all_test2_loss' + ratio + '.npy', allow_pickle=True)
        # ############## the average of t min ###################
        t = 5
        all_val1_amin = all_val1_loss
        all_test1_amin = all_test1_loss

        all_a_amin = all_val2_loss[:3276]
        all_b_amin = all_val2_loss[3276:14084]
        all_c_amin = all_val2_loss[14084:14280]
        all_d_amin = all_val2_loss[14280:15036]
        # all_f_amin = all_val2_loss[15036:15988]  # 15036:17276
        all_f_amin = all_val2_loss[15036:17276]
        all_other_amin = all_val2_loss

        all_ab_amin = all_test2_loss[:8176]  # 8176
        all_bb_amin = all_test2_loss[8176:11088]
        all_cb_amin = all_test2_loss[11088:11760]
        all_db_amin = all_test2_loss[11760:12544]
        all_fb_amin = all_test2_loss[12544:13496]
        all_otherb_amin = all_test2_loss

        all_val1_amin = [sorted(x, reverse=False) for x in chunked(all_val1_amin, frame_len)]
        all_a_amin = [sorted(x, reverse=False) for x in chunked(all_a_amin, frame_len)]
        all_b_amin = [sorted(x, reverse=False) for x in chunked(all_b_amin, frame_len)]
        all_c_amin = [sorted(x, reverse=False) for x in chunked(all_c_amin, frame_len)]
        all_d_amin = [sorted(x, reverse=False) for x in chunked(all_d_amin, frame_len)]
        all_f_amin = [sorted(x, reverse=False) for x in chunked(all_f_amin, frame_len)]
        all_other_amin = [sorted(x, reverse=False) for x in chunked(all_other_amin, frame_len)]

        all_test1_amin = [sorted(x, reverse=False) for x in chunked(all_test1_amin, frame_len)]
        all_ab_amin = [sorted(x, reverse=False) for x in chunked(all_ab_amin, frame_len)]
        all_bb_amin = [sorted(x, reverse=False) for x in chunked(all_bb_amin, frame_len)]
        all_cb_amin = [sorted(x, reverse=False) for x in chunked(all_cb_amin, frame_len)]
        all_db_amin = [sorted(x, reverse=False) for x in chunked(all_db_amin, frame_len)]
        all_fb_amin = [sorted(x, reverse=False) for x in chunked(all_fb_amin, frame_len)]
        all_otherb_amin = [sorted(x, reverse=False) for x in chunked(all_otherb_amin, frame_len)]

        all_val1_amin = np.array([sum(x[:t]) for x in all_val1_amin])
        all_a_amin = np.array([sum(x[:t]) for x in all_a_amin])
        all_b_amin = np.array([sum(x[:t]) for x in all_b_amin])
        all_c_amin = np.array([sum(x[:t]) for x in all_c_amin])
        all_d_amin = np.array([sum(x[:t]) for x in all_d_amin])
        all_f_amin = np.array([sum(x[:t]) for x in all_f_amin])
        all_other_amin = np.array([sum(x[:t]) for x in all_other_amin])

        all_test1_amin = np.array([sum(x[:t]) for x in all_test1_amin])
        all_ab_amin = np.array([sum(x[:t]) for x in all_ab_amin])
        all_bb_amin = np.array([sum(x[:t]) for x in all_bb_amin])
        all_cb_amin = np.array([sum(x[:t]) for x in all_cb_amin])
        all_db_amin = np.array([sum(x[:t]) for x in all_db_amin])
        all_fb_amin = np.array([sum(x[:t]) for x in all_fb_amin])
        all_otherb_amin = np.array([sum(x[:t]) for x in all_otherb_amin])

        y_true_e_amin = [0] * all_val1_amin.shape[0] + [1] * all_test1_amin.shape[0]
        y_pred_e_amin = np.concatenate((all_val1_amin, all_test1_amin), axis=0)
        y_pred_e_amin = np.array(y_pred_e_amin)

        y_true_a_amin = [0] * all_a_amin.shape[0] + [1] * all_ab_amin.shape[0]
        y_pred_a_amin = np.concatenate((all_a_amin, all_ab_amin), axis=0)
        y_pred_a_amin = np.array(y_pred_a_amin)

        y_true_b_amin = [0] * all_b_amin.shape[0] + [1] * all_bb_amin.shape[0]
        y_pred_b_amin = np.concatenate((all_b_amin, all_bb_amin), axis=0)
        y_pred_b_amin = np.array(y_pred_b_amin)

        y_true_c_amin = [0] * all_c_amin.shape[0] + [1] * all_cb_amin.shape[0]
        y_pred_c_amin = np.concatenate((all_c_amin, all_cb_amin), axis=0)
        y_pred_c_amin = np.array(y_pred_c_amin)

        y_true_d_amin = [0] * all_d_amin.shape[0] + [1] * all_db_amin.shape[0]
        y_pred_d_amin = np.concatenate((all_d_amin, all_db_amin), axis=0)
        y_pred_d_amin = np.array(y_pred_d_amin)

        y_true_f_amin = [0] * all_f_amin.shape[0] + [1] * all_fb_amin.shape[0]
        y_pred_f_amin = np.concatenate((all_f_amin, all_fb_amin), axis=0)
        y_pred_f_amin = np.array(y_pred_f_amin)

        y_true_other_amin = [0] * all_other_amin.shape[0] + [1] * all_otherb_amin.shape[0]
        y_pred_other_amin = np.concatenate((all_other_amin, all_otherb_amin), axis=0)
        y_pred_other_amin = np.array(y_pred_other_amin)

        auc_e_amin = roc_auc_score(y_true_e_amin, y_pred_e_amin)
        auc_a_amin = roc_auc_score(y_true_a_amin, y_pred_a_amin)
        auc_b_amin = roc_auc_score(y_true_b_amin, y_pred_b_amin)
        auc_c_amin = roc_auc_score(y_true_c_amin, y_pred_c_amin)
        auc_d_amin = roc_auc_score(y_true_d_amin, y_pred_d_amin)
        auc_f_amin = roc_auc_score(y_true_f_amin, y_pred_f_amin)
        auc_other_amin = roc_auc_score(y_true_other_amin, y_pred_other_amin)
        auc_e_loss_amin.append(auc_e_amin)
        auc_a_loss_amin.append(auc_a_amin)
        auc_b_loss_amin.append(auc_b_amin)
        auc_c_loss_amin.append(auc_c_amin)
        auc_d_loss_amin.append(auc_d_amin)
        auc_f_loss_amin.append(auc_f_amin)
        auc_other_loss_amin.append(auc_other_amin)

        pauc_e_amin = roc_auc_score(y_true_e_amin, y_pred_e_amin, max_fpr=0.1)
        pauc_a_amin = roc_auc_score(y_true_a_amin, y_pred_a_amin, max_fpr=0.1)
        pauc_b_amin = roc_auc_score(y_true_b_amin, y_pred_b_amin, max_fpr=0.1)
        pauc_c_amin = roc_auc_score(y_true_c_amin, y_pred_c_amin, max_fpr=0.1)
        pauc_d_amin = roc_auc_score(y_true_d_amin, y_pred_d_amin, max_fpr=0.1)
        pauc_f_amin = roc_auc_score(y_true_f_amin, y_pred_f_amin, max_fpr=0.1)
        pauc_other_amin = roc_auc_score(y_true_other_amin, y_pred_other_amin, max_fpr=0.1)
        pauc_e_loss_amin.append(pauc_e_amin)
        pauc_a_loss_amin.append(pauc_a_amin)
        pauc_b_loss_amin.append(pauc_b_amin)
        pauc_c_loss_amin.append(pauc_c_amin)
        pauc_d_loss_amin.append(pauc_d_amin)
        pauc_f_loss_amin.append(pauc_f_amin)
        pauc_other_loss_amin.append(pauc_other_amin)

    return auc_e_loss_amin, auc_a_loss_amin, auc_b_loss_amin, auc_c_loss_amin, auc_d_loss_amin, auc_f_loss_amin, auc_other_loss_amin, \
        pauc_e_loss_amin, pauc_a_loss_amin, pauc_b_loss_amin, pauc_c_loss_amin, pauc_d_loss_amin, pauc_f_loss_amin, pauc_other_loss_amin


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


def main():
    # for ratio in ratio_list:
    #     training_loss_plot = np.load('data_only_e/train_loss' + ratio + '.npy', allow_pickle=True)
    #     test_ae_normal_plot = np.load('data_only_e/test_ae_normal_loss' + ratio + '.npy', allow_pickle=True)
    #     test_ae_abnormal_plot = np.load('data_only_e/test_ae_abnormal_loss' + ratio + '.npy', allow_pickle=True)
    #     test_other_normal_plot = np.load('data_only_e/test_other_normal_loss' + ratio + '.npy', allow_pickle=True)
    #     test_other_abnormal_plot = np.load('data_only_e/test_other_abnormal_loss' + ratio + '.npy', allow_pickle=True)
    #
    #     training_kl_loss_plot = np.load('data_only_e/train_kl_loss' + ratio + '.npy', allow_pickle=True)
    #     test_ae_normal_kl_plot = np.load('data_only_e/test_ae_normal_kl_loss' + ratio + '.npy', allow_pickle=True)
    #     test_ae_abnormal_kl_plot = np.load('data_only_e/test_ae_abnormal_kl_loss' + ratio + '.npy', allow_pickle=True)
    #     test_other_normal_kl_plot = np.load('data_only_e/test_other_normal_kl_loss' + ratio + '.npy', allow_pickle=True)
    #     test_other_abnormal_kl_plot = np.load('data_only_e/test_other_abnormal_kl_loss' + ratio + '.npy', allow_pickle=True)
    #
    #     auc_ae_ave, auc_other_ave, auc_ae_min, auc_other_min, auc_ae_amin, auc_other_amin, auc_ae_max, auc_other_max, auc_ae_amax, auc_other_amax = Loss_AUC(ratio)
    #     plot_loss(ratio,
    #               training_loss_plot,
    #               test_ae_normal_plot,
    #               test_ae_abnormal_plot,
    #               test_other_normal_plot,
    #               test_other_abnormal_plot,
    #               auc_ae_ave,
    #               auc_other_ave,
    #               auc_ae_amin,
    #               auc_other_amin,
    #               auc_ae_amax,
    #               auc_other_amax)
    #
    #     plot_kl_loss(ratio,
    #                  training_kl_loss_plot,
    #                  test_ae_normal_kl_plot,
    #                  test_ae_abnormal_kl_plot,
    #                  test_other_normal_kl_plot,
    #                  test_other_abnormal_kl_plot,
    #                  auc_ae_ave,
    #                  auc_other_ave,
    #                  auc_ae_amin,
    #                  auc_other_amin,
    #                  auc_ae_amax,
    #                  auc_other_amax)

    print('\nloss')
    for ratio in ratio_list:
        print('***********************')
        print(ratio)
        auc_e_loss_amin, auc_a_loss_amin, auc_b_loss_amin, auc_c_loss_amin, auc_d_loss_amin, auc_f_loss_amin, auc_other_loss_amin, \
        pauc_e_loss_amin, pauc_a_loss_amin, pauc_b_loss_amin, pauc_c_loss_amin, pauc_d_loss_amin, pauc_f_loss_amin, pauc_other_loss_amin\
            = Loss_AUC(ratio)
        print('********* amin **********')
        print(auc_e_loss_amin)
        print(pauc_e_loss_amin)
        print(auc_a_loss_amin)
        print(pauc_a_loss_amin)
        print(auc_b_loss_amin)
        print(pauc_b_loss_amin)
        print(auc_c_loss_amin)
        print(pauc_c_loss_amin)
        print(auc_d_loss_amin)
        print(pauc_d_loss_amin)
        print(auc_f_loss_amin)
        print(pauc_f_loss_amin)
        print(pauc_other_loss_amin)
        sss = np.argmax(auc_e_loss_amin)
        print('auc_e:', (max(auc_e_loss_amin)), 'epoch:', np.argmax(auc_e_loss_amin))
        print('pauc_e:', (max(pauc_e_loss_amin)), 'epoch:', np.argmax(pauc_e_loss_amin))

        print('auc_a:', auc_a_loss_amin[sss.real])
        print('pauc_a:', pauc_a_loss_amin[sss.real])
        print('auc_b:', auc_b_loss_amin[sss.real])
        print('pauc_b:', pauc_b_loss_amin[sss.real])
        print('auc_c:', auc_c_loss_amin[sss.real])
        print('pauc_c:', pauc_c_loss_amin[sss.real])
        print('auc_d:', auc_d_loss_amin[sss.real])
        print('pauc_d:', pauc_d_loss_amin[sss.real])
        print('auc_f:', auc_f_loss_amin[sss.real])
        print('pauc_f:', pauc_f_loss_amin[sss.real])
        print('auc_other:', auc_other_loss_amin[sss.real])
        print('pauc_other:', pauc_other_loss_amin[sss.real])
        # best
        print('\nbest*****************')
        print('auc_a:', (max(auc_a_loss_amin)), 'epoch:', np.argmax(auc_a_loss_amin))
        print('pauc_a:', (max(pauc_a_loss_amin)), 'epoch:', np.argmax(pauc_a_loss_amin))
        print('auc_b:', (max(auc_b_loss_amin)), 'epoch:', np.argmax(auc_b_loss_amin))
        print('pauc_b:', (max(pauc_b_loss_amin)), 'epoch:', np.argmax(pauc_b_loss_amin))
        print('auc_c:', (max(auc_c_loss_amin)), 'epoch:', np.argmax(auc_c_loss_amin))
        print('pauc_c:', (max(pauc_c_loss_amin)), 'epoch:', np.argmax(pauc_c_loss_amin))
        print('auc_d:', (max(auc_d_loss_amin)), 'epoch:', np.argmax(auc_d_loss_amin))
        print('pauc_d:', (max(pauc_d_loss_amin)), 'epoch:', np.argmax(pauc_d_loss_amin))
        print('auc_f:', (max(auc_f_loss_amin)), 'epoch:', np.argmax(auc_f_loss_amin))
        print('pauc_f:', (max(pauc_f_loss_amin)), 'epoch:', np.argmax(pauc_f_loss_amin))
        print('auc_other:', (max(auc_other_loss_amin)), 'epoch:', np.argmax(auc_other_loss_amin))
        print('pauc_other:', (max(pauc_other_loss_amin)), 'epoch:', np.argmax(pauc_other_loss_amin))

    # print('\nlatent')
    # for ratio in ratio_list:
    #     print('***********************')
    #     print(ratio)
    #     auc, pauc, auc_other, pauc_other = Latent_AUC(ratio)
    #     print('auc:', auc)
    #     print('pauc:', pauc)
    #     print('auc_e:', (max(auc)), 'epoch:', np.argmax(auc))
    #     print('pauc_e:', (max(pauc)), 'epoch:', np.argmax(pauc))
    #     print('auc_other:', auc_other)
    #     print('pauc_other:', pauc_other)
    #     print('auc_e_other:', (max(auc_other)), 'epoch:', np.argmax(auc_other))
    #     print('pauc_e_other:', (max(pauc_other)), 'epoch:', np.argmax(pauc_other))
    #
    # print('\nkl_loss')
    # for ratio in ratio_list:
    #     if ratio == 'AE':
    #         pass
    #     else:
    #         print('***********************')
    #         print(ratio)
    #         auc, pauc, auc_other, pauc_other = kl_AUC(ratio)
    #         print('auc:', auc)
    #         print('pauc:', pauc)
    #         print('auc_e:', (max(auc)), 'epoch:', np.argmax(auc))
    #         print('pauc_e:', (max(pauc)), 'epoch:', np.argmax(pauc))
    #         print('auc_other:', auc_other)
    #         print('pauc_other:', pauc_other)
    #         print('auc_e_other:', (max(auc_other)), 'epoch:', np.argmax(auc_other))
    #         print('pauc_e_other:', (max(pauc_other)), 'epoch:', np.argmax(pauc_other))


if __name__ == '__main__':
    main()
    # print_test_loss()

