import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import mode
import seaborn as sns
import os
from pylab import mpl
from more_itertools import chunked

epoch_list = [15, 15, 18, 15, 15, 15]
# ratio_list = ['1.00-1.00', '0.10-1.00', '0.01-1.00']
ratio_list = ['0.00-1.00', '0.01-1.00', '0.10-1.00', '1.00-1.00', '10.00-1.00', '100.00-1.00']
data_type = '_all'

# def Post_processing():
#     auc_ae_loss_ave = []
#     auc_other_loss_ave = []
#     auc_ae_loss_max = []
#     auc_other_loss_max = []
#     auc_ae_loss_amax = []
#     auc_other_loss_amax = []
#     auc_ae_loss_min = []
#     auc_other_loss_min = []
#     auc_ae_loss_amin = []
#     auc_other_loss_amin = []
#     for i in range(EPOCH_MAX):
#         # ############## average ################
#         all_val1_ave = all_val1_loss
#         all_val2_ave = all_val2_loss
#         all_test1_ave = all_test1_loss
#         all_test2_ave = all_test2_loss
#
#         # all_val1_ave = np.array([sum(x) / len(x) for x in chunked(all_val1_ave, 28)])
#         # all_val2_ave = np.array([sum(x) / len(x) for x in chunked(all_val2_ave, 28)])
#         # all_test1_ave = np.array([sum(x) / len(x) for x in chunked(all_test1_ave, 28)])
#         # all_test2_ave = np.array([sum(x) / len(x) for x in chunked(all_test2_ave, 28)])
#         all_val1_ave = np.array([sum(x) for x in chunked(all_val1_ave, 28)])
#         all_val2_ave = np.array([sum(x) for x in chunked(all_val2_ave, 28)])
#         all_test1_ave = np.array([sum(x) for x in chunked(all_test1_ave, 28)])
#         all_test2_ave = np.array([sum(x) for x in chunked(all_test2_ave, 28)])
#
#         y_true_ae_ave = [0] * all_val1_ave.shape[0] + [1] * all_test1_ave.shape[0]
#         y_pred_ae_ave = np.concatenate((all_val1_ave, all_test1_ave), axis=0)
#         y_pred_ae_ave = np.array(y_pred_ae_ave)
#         y_true_other_ave = [0] * all_val2_ave.shape[0] + [1] * all_test2_ave.shape[0]
#         y_pred_other_ave = np.concatenate((all_val2_ave, all_test2_ave), axis=0)
#         y_pred_other_ave = np.array(y_pred_other_ave)
#
#         auc_ae_ave = roc_auc_score(y_true_ae_ave, y_pred_ae_ave)
#         auc_other_ave = roc_auc_score(y_true_other_ave, y_pred_other_ave)
#         auc_ae_loss_ave.append(auc_ae_ave)
#         auc_other_loss_ave.append(auc_other_ave)
#         # ############## min ####################
#         all_val1_min = all_val1_loss
#         all_val2_min = all_val2_loss
#         all_test1_min = all_test1_loss
#         all_test2_min = all_test2_loss
#
#         # all_val1_max = np.array([max(x) for x in chunked(all_val1_max, 28)])
#         # all_val2_max = np.array([max(x) for x in chunked(all_val2_max, 28)])
#         # all_test1_max = np.array([max(x) for x in chunked(all_test1_max, 28)])
#         # all_test2_max = np.array([max(x) for x in chunked(all_test2_max, 28)])
#         all_val1_min = np.array([min(x) for x in chunked(all_val1_min, 28)])
#         all_val2_min = np.array([min(x) for x in chunked(all_val2_min, 28)])
#         all_test1_min = np.array([min(x) for x in chunked(all_test1_min, 28)])
#         all_test2_min = np.array([min(x) for x in chunked(all_test2_min, 28)])
#
#         y_true_ae_min = [0] * all_val1_min.shape[0] + [1] * all_test1_min.shape[0]
#         y_pred_ae_min = np.concatenate((all_val1_min, all_test1_min), axis=0)
#         y_pred_ae_min = np.array(y_pred_ae_min)
#         y_true_other_min = [0] * all_val2_min.shape[0] + [1] * all_test2_min.shape[0]
#         y_pred_other_min = np.concatenate((all_val2_min, all_test2_min), axis=0)
#         y_pred_other_min = np.array(y_pred_other_min)
#
#         auc_ae_min = roc_auc_score(y_true_ae_min, y_pred_ae_min)
#         auc_other_min = roc_auc_score(y_true_other_min, y_pred_other_min)
#         auc_ae_loss_min.append(auc_ae_min)
#         auc_other_loss_min.append(auc_other_min)
#         # ############## the average of t min ###################
#         t = 5
#         all_val1_amin = all_val1_loss
#         all_val2_amin = all_val2_loss
#         all_test1_amin = all_test1_loss
#         all_test2_amin = all_test2_loss
#
#         all_val1_amin = [sorted(x, reverse=False) for x in chunked(all_val1_amin, 28)]
#         all_val2_amin = [sorted(x, reverse=False) for x in chunked(all_val2_amin, 28)]
#         all_test1_amin = [sorted(x, reverse=False) for x in chunked(all_test1_amin, 28)]
#         all_test2_amin = [sorted(x, reverse=False) for x in chunked(all_test2_amin, 28)]
#         all_val1_amin = np.array([sum(x[:t]) for x in all_val1_amin])
#         all_val2_amin = np.array([sum(x[:t]) for x in all_val2_amin])
#         all_test1_amin = np.array([sum(x[:t]) for x in all_test1_amin])
#         all_test2_amin = np.array([sum(x[:t]) for x in all_test2_amin])
#
#         # k1 = 10
#         # k2 = 16
#         # all_val1_amin = np.array([sum(x[k1:k2]) for x in all_val1_amin])
#         # all_val2_amin = np.array([sum(x[k1:k2]) for x in all_val2_amin])
#         # all_test1_amin = np.array([sum(x[k1:k2]) for x in all_test1_amin])
#         # all_test2_amin = np.array([sum(x[k1:k2]) for x in all_test2_amin])
#
#         y_true_ae_amin = [0] * all_val1_amin.shape[0] + [1] * all_test1_amin.shape[0]
#         y_pred_ae_amin = np.concatenate((all_val1_amin, all_test1_amin), axis=0)
#         y_pred_ae_amin = np.array(y_pred_ae_amin)
#         y_true_other_amin = [0] * all_val2_amin.shape[0] + [1] * all_test2_amin.shape[0]
#         y_pred_other_amin = np.concatenate((all_val2_amin, all_test2_amin), axis=0)
#         y_pred_other_amin = np.array(y_pred_other_amin)
#
#         auc_ae_amin = roc_auc_score(y_true_ae_amin, y_pred_ae_amin)
#         auc_other_amin = roc_auc_score(y_true_other_amin, y_pred_other_amin)
#         auc_ae_loss_amin.append(auc_ae_amin)
#         auc_other_loss_amin.append(auc_other_amin)
#         # ############## max ####################
#         all_val1_max = all_val1_loss
#         all_val2_max = all_val2_loss
#         all_test1_max = all_test1_loss
#         all_test2_max = all_test2_loss
#
#         all_val1_max = np.array([max(x) for x in chunked(all_val1_max, 28)])
#         all_val2_max = np.array([max(x) for x in chunked(all_val2_max, 28)])
#         all_test1_max = np.array([max(x) for x in chunked(all_test1_max, 28)])
#         all_test2_max = np.array([max(x) for x in chunked(all_test2_max, 28)])
#
#         y_true_ae_max = [0] * all_val1_max.shape[0] + [1] * all_test1_max.shape[0]
#         y_pred_ae_max = np.concatenate((all_val1_max, all_test1_max), axis=0)
#         y_pred_ae_max = np.array(y_pred_ae_max)
#         y_true_other_max = [0] * all_val2_max.shape[0] + [1] * all_test2_max.shape[0]
#         y_pred_other_max = np.concatenate((all_val2_max, all_test2_max), axis=0)
#         y_pred_other_max = np.array(y_pred_other_max)
#
#         auc_ae_max = roc_auc_score(y_true_ae_max, y_pred_ae_max)
#         auc_other_max = roc_auc_score(y_true_other_max, y_pred_other_max)
#         auc_ae_loss_max.append(auc_ae_max)
#         auc_other_loss_max.append(auc_other_max)
#         # ############## the average of t max ###################
#         all_val1_amax = all_val1_loss
#         all_val2_amax = all_val2_loss
#         all_test1_amax = all_test1_loss
#         all_test2_amax = all_test2_loss
#
#         all_val1_amax = [sorted(x, reverse=True) for x in chunked(all_val1_amax, 28)]
#         all_val2_amax = [sorted(x, reverse=True) for x in chunked(all_val2_amax, 28)]
#         all_test1_amax = [sorted(x, reverse=True) for x in chunked(all_test1_amax, 28)]
#         all_test2_amax = [sorted(x, reverse=True) for x in chunked(all_test2_amax, 28)]
#         all_val1_amax = np.array([sum(x[:t]) for x in all_val1_amax])
#         all_val2_amax = np.array([sum(x[:t]) for x in all_val2_amax])
#         all_test1_amax = np.array([sum(x[:t]) for x in all_test1_amax])
#         all_test2_amax = np.array([sum(x[:t]) for x in all_test2_amax])
#
#         y_true_ae_amax = [0] * all_val1_amax.shape[0] + [1] * all_test1_amax.shape[0]
#         y_pred_ae_amax = np.concatenate((all_val1_amax, all_test1_amax), axis=0)
#         y_pred_ae_amax = np.array(y_pred_ae_amax)
#         y_true_other_amax = [0] * all_val2_amax.shape[0] + [1] * all_test2_amax.shape[0]
#         y_pred_other_amax = np.concatenate((all_val2_amax, all_test2_amax), axis=0)
#         y_pred_other_amax = np.array(y_pred_other_amax)
#
#         auc_ae_amax = roc_auc_score(y_true_ae_amax, y_pred_ae_amax)
#         auc_other_amax = roc_auc_score(y_true_other_amax, y_pred_other_amax)
#         auc_ae_loss_amax.append(auc_ae_amax)
#         auc_other_loss_amax.append(auc_other_amax)
#
#     return auc_ae_loss_ave, auc_other_loss_ave, \
#            auc_ae_loss_min, auc_other_loss_min, \
#            auc_ae_loss_amin, auc_other_loss_amin, \
#            auc_ae_loss_max, auc_other_loss_max, \
#            auc_ae_loss_amax, auc_other_loss_amax

for i in range(6):
    print(data_type)
    print(ratio_list[i])
    # loss
    all_val1_loss = np.load('clustering' + data_type + '/epoch/' + str(epoch_list[i]) + '/all_val1_loss' + ratio_list[i] + '.npy', allow_pickle=True)
    all_test1_loss = np.load('clustering' + data_type + '/epoch/' + str(epoch_list[i]) + '/all_test1_loss' + ratio_list[i] + '.npy', allow_pickle=True)
    # # ues amin loss
    # t = 5
    # frame_len = 28
    # all_val1_ave = [sorted(x, reverse=False) for x in chunked(all_val1_loss, frame_len)]
    # all_test1_ave = [sorted(x, reverse=False) for x in chunked(all_test1_loss, frame_len)]
    # all_val1_ave = np.array([sum(x[:t]) for x in all_val1_ave])
    # all_test1_ave = np.array([sum(x[:t]) for x in all_test1_ave])
    # ues average loss
    all_val1_ave = np.array([sum(x) for x in chunked(all_val1_loss, 28)])
    all_test1_ave = np.array([sum(x) for x in chunked(all_test1_loss, 28)])

    # weights1 = np.ones_like(all_val1_ave)/float(len(all_val1_ave))
    # weights2 = np.ones_like(all_test1_ave)/float(len(all_test1_ave))
    # print('\n' + ratio_list[i])
    # print('loss')
    # print('mean:', all_val1_ave.mean(), all_test1_ave.mean())
    # print('std:', all_val1_ave.std(), all_test1_ave.std())
    # print('*********************************************************')
    # fig = plt.figure()
    # ax1 = fig.add_subplot()
    # ax2 = ax1.twinx()
    # # ax1.hist(all_val1_ave, weights=weights1, bins=100, label='normal')
    # # ax2.hist(all_test1_ave, weights=weights2, bins=100, label='abnormal', color='r')
    # ax1.hist(all_val1_ave, bins=200, range=(0, 2), label='normal', histtype='stepfilled', density=True)
    # ax2.hist(all_test1_ave, bins=200, range=(0, 2), label='abnormal', histtype='stepfilled', density=True, color='r')
    # # x轴区间范围
    # plt.xlabel('LOSS')
    # plt.xlim(0, 2)
    # # ax1.set_ylim(0, 0.3)
    # # ax2.set_ylim(0, 0.3)
    # ax1.set_ylabel('normal')
    # ax2.set_ylabel('abnormal')
    # ax1.legend(loc='upper left')
    # ax2.legend(loc='upper right')
    # plt.title('Normal sample frequency distribution histogram of E dataset')
    # plt.savefig('clustering_only_e/loss_histogram_learning_rate_0.0001_' + ratio_list[i] + '.png')
    # plt.show()
    #

    #  kl_loss
    all_val1_kl_loss_ave = np.load('clustering' + data_type + '/epoch/' + str(epoch_list[i]) + '/all_val1_kl_loss' + ratio_list[i] + '.npy', allow_pickle=True)
    all_test1_kl_loss_ave = np.load('clustering' + data_type + '/epoch/' + str(epoch_list[i]) + '/all_test1_kl_loss' + ratio_list[i] + '.npy', allow_pickle=True)
    # weights1 = np.ones_like(all_val1_kl_loss_ave)/float(len(all_val1_kl_loss_ave))
    # weights2 = np.ones_like(all_test1_kl_loss_ave)/float(len(all_test1_kl_loss_ave))
    # print('kl_loss')
    # print('mean:', all_val1_kl_loss_ave.mean(), all_test1_kl_loss_ave.mean())
    # print('std:', all_val1_kl_loss_ave.std(), all_test1_kl_loss_ave.std())
    # print('*********************************************************')
    # fig = plt.figure()
    # ax1 = fig.add_subplot()
    # ax2 = ax1.twinx()
    # # ax1.hist(all_val1_kl_loss_ave, weights=weights1, bins=100, label='normal')
    # # ax2.hist(all_test1_kl_loss_ave, weights=weights2, bins=100, label='abnormal', color='r')
    # ax1.hist(all_val1_kl_loss_ave, bins=200, range=(0, 2), label='normal', histtype='stepfilled', density=True)
    # ax2.hist(all_test1_kl_loss_ave, bins=200, range=(0, 2), label='abnormal', histtype='stepfilled', density=True, color='r')
    # # x轴区间范围
    # plt.xlabel('LOSS')
    # plt.xlim(0, 2)
    # # ax1.set_ylim(0, 0.4)
    # # ax2.set_ylim(0, 0.4)
    # ax1.set_ylabel('normal')
    # ax2.set_ylabel('abnormal')
    # ax1.legend(loc='upper left')
    # ax2.legend(loc='upper right')
    # plt.title('Normal sample frequency distribution histogram of E dataset')
    # plt.savefig('clustering_only_e/kl_histogram_learning_rate_0.0001_' + ratio_list[i] + '.png')
    # plt.show()

    # 原始数据
    x1 = np.concatenate((all_val1_ave, all_test1_ave), axis=0)
    y1 = np.concatenate((all_val1_kl_loss_ave, all_test1_kl_loss_ave), axis=0)
    # x1 = all_val1_ave
    # y1 = all_val1_kl_loss_ave
    X1 = pd.Series(x1)
    Y1 = pd.Series(y1)

    # 处理数据删除Nan
    x1 = X1.dropna()
    y1 = Y1.dropna()
    n = x1.count()
    x1.index = np.arange(n)
    y1.index = np.arange(n)

    # # 分部计算
    # d = (x1.sort_values().index - y1.sort_values().index) ** 2
    # dd = d.to_series().sum()
    # p = 1 - n * dd / (n * (n ** 2 - 1))
    # s.corr()函数计算

    spearman = x1.corr(y1, method='spearman')
    print('spearman: s: {:.4f}'.format(spearman))  # 0.942857142857143 0.9428571428571428
    pearson = X1.corr(Y1, method="pearson")  # 皮尔森相关性系数 #0.948136664010285
    print('pearson: p: {:.4f}'.format(pearson))  # 0.942857142857143 0.9428571428571428

