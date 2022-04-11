import numpy as np
from scipy.stats import norm
from scipy.stats import normaltest
from scipy.stats import kstest
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn import mixture
import pandas as pd
import operator
from functools import reduce
from more_itertools import chunked

np.random.seed(128)  # 128

a = 0  # 数学期望
b = 1  # 方差
x = np.random.randn(10000, 16)  # 正态分布
gmm = mixture.GaussianMixture()
gmm.fit(x)  # 用EM算法估计模型参数

# a = gmm.covariances_
y = np.random.rand(10000, 16)  # 均匀分布
k = np.random.randn(10000, 16)  # 正态分布

print('均匀分布：', gmm.score(y))
print('正态分布：', gmm.score(k))
dict_density_ratio = {
'_a_a_density_ratio': {'0.01-1.00': 26, '0.10-1.00': 25, '1.00-1.00': 27, '10.00-1.00': 23, '100.00-1.00': 32},
'_b_a_density_ratio': {'0.01-1.00': 1,  '0.10-1.00': 6,  '1.00-1.00': 6,  '10.00-1.00': 1, '100.00-1.00': 4},
'_d_a_density_ratio': {'0.01-1.00': 12, '0.10-1.00': 39, '1.00-1.00': 39, '10.00-1.00': 12, '100.00-1.00': 16},
'_e_a_density_ratio': {'0.01-1.00': 11, '0.10-1.00': 18, '1.00-1.00': 14, '10.00-1.00': 0, '100.00-1.00': 19},
'_f_a_density_ratio': {'0.01-1.00': 41, '0.10-1.00': 35, '1.00-1.00': 35, '10.00-1.00': 34, '100.00-1.00': 14},

'_ae_a_density_ratio': {'0.01-1.00': 17, '0.10-1.00': 41, '1.00-1.00': 49, '10.00-1.00': 21, '100.00-1.00': 19},
'_ef_a_density_ratio': {'0.01-1.00': 15, '0.10-1.00': 15, '1.00-1.00': 48, '10.00-1.00': 1, '100.00-1.00': 2},
'_all_a_density_ratio': {'0.01-1.00': 13, '0.10-1.00': 13, '1.00-1.00': 49, '10.00-1.00': 11, '100.00-1.00': 24},

'_m_a_density_ratio': {'0.01-1.00': 45, '0.10-1.00': 43, '1.00-1.00': 47, '10.00-1.00': 30, '100.00-1.00': 43},


                      }
dict_density_AE = {
'_a_a_density_AE': {'0.01-1.00': 14, '0.10-1.00': 9,  '1.00-1.00': 28, '10.00-1.00': 15, '100.00-1.00': 9},
'_b_a_density_AE': {'0.01-1.00': 2,  '0.10-1.00': 23, '1.00-1.00': 1,  '10.00-1.00': 2, '100.00-1.00': 16},
'_d_a_density_AE': {'0.01-1.00': 15, '0.10-1.00': 39, '1.00-1.00': 15, '10.00-1.00': 35, '100.00-1.00': 17},
'_e_a_density_AE': {'0.01-1.00': 6, '0.10-1.00': 8, '1.00-1.00': 14, '10.00-1.00': 13, '100.00-1.00': 12},
'_f_a_density_AE': {'0.01-1.00': 24, '0.10-1.00': 21, '1.00-1.00': 16, '10.00-1.00': 19, '100.00-1.00': 16},

'_ae_a_density_AE': {'0.01-1.00': 14, '0.10-1.00': 13, '1.00-1.00': 18, '10.00-1.00': 34, '100.00-1.00': 49},
'_ef_a_density_AE': {'0.01-1.00': 14, '0.10-1.00': 14, '1.00-1.00': 14, '10.00-1.00': 14, '100.00-1.00': 11},
'_all_a_density_AE': {'0.01-1.00': 13, '0.10-1.00': 14, '1.00-1.00': 12, '10.00-1.00': 14, '100.00-1.00': 48},

'_m_a_density_AE': {'0.01-1.00': 31, '0.10-1.00': 49, '1.00-1.00': 49, '10.00-1.00': 0, '100.00-1.00': 49},
                      }
dict_only = {
'_only_a': {'0.01-1.00': 8, '0.10-1.00': 23,  '1.00-1.00': 6, '10.00-1.00': 16, '100.00-1.00': 3},
'_only_b': {'0.01-1.00': 7,  '0.10-1.00': 1, '1.00-1.00': 1,  '10.00-1.00': 1, '100.00-1.00': 40},
'_only_d': {'0.01-1.00': 20, '0.10-1.00': 9, '1.00-1.00': 20, '10.00-1.00': 42, '100.00-1.00': 23},
'_only_e': {'0.01-1.00': 14, '0.10-1.00': 16, '1.00-1.00': 1, '10.00-1.00': 1, '100.00-1.00': 1},
'_only_f': {'0.01-1.00': 4, '0.10-1.00': 5, '1.00-1.00': 4, '10.00-1.00': 13, '100.00-1.00': 3},

'_ae': {'0.01-1.00': 15, '0.10-1.00': 18, '1.00-1.00': 15, '10.00-1.00': 15, '100.00-1.00': 15},
'_ef': {'0.01-1.00': 13, '0.10-1.00': 16, '1.00-1.00': 7, '10.00-1.00': 5, '100.00-1.00': 7},
'_all': {'0.01-1.00': 10, '0.10-1.00': 49, '1.00-1.00': 13, '10.00-1.00': 14, '100.00-1.00': 16},

'_only_m_a': {'0.01-1.00': 45, '0.10-1.00': 43, '1.00-1.00': 47, '10.00-1.00': 30, '100.00-1.00': 45},
                      }

data_type1 = ['_a_a_density_ratio', '_b_a_density_ratio', '_d_a_density_ratio', '_e_a_density_ratio', '_f_a_density_ratio', '_ae_a_density_ratio', '_ef_a_density_ratio', '_all_a_density_ratio', '_m_a_density_ratio']
data_type2 = ['_a_a_density_AE', '_b_a_density_AE', '_d_a_density_AE', '_e_a_density_AE', '_f_a_density_AE', '_ae_a_density_AE', '_ef_a_density_AE', '_all_a_density_AE', '_m_a_density_AE']
data_type3 = ['_only_a', '_only_b', '_only_d', '_only_e', '_only_f', '_ae', '_ef', '_all', '_only_m_a']
ratio = ['0.01-1.00', '0.10-1.00', '1.00-1.00', '10.00-1.00', '100.00-1.00']
# ratio = ['10.00-1.00']


def alys(type1, type2, type3, ratio1):
    print('\n' + ratio1)
    latent1 = np.load('clustering' + type1 + '/''/epoch/' + str(dict_density_ratio[type1][ratio1]) + '/all_training_latent' + ratio1 + '.npy', allow_pickle=True)
    latent2 = np.load('clustering' + type2 + '/''/epoch/' + str(dict_density_AE[type2][ratio1]) + '/all_training_latent' + ratio1 + '.npy', allow_pickle=True)
    latent3 = np.load('clustering' + type3 + '/''/epoch/' + str(dict_only[type3][ratio1]) + '/all_training_latent' + ratio1 + '.npy', allow_pickle=True)

    # tsne = manifold.TSNE(n_components=1, init='pca', random_state=999)
    # X_tsne1 = tsne.fit_transform(latent1)
    # X_tsne2 = tsne.fit_transform(latent2)
    # print('_d_a_density_AE', norm.fit(latent1))  # 返回极大似然估计，估计出参数约为30和2
    # print('_only_d', norm.fit(latent2))  # 返回极大似然估计，估计出参数约为30和2
    # ########################## 分析数据与标准正态分布的相似性 ###########################################
    # S1, P1 = normaltest(latent1)
    # S2, P2 = normaltest(latent2)
    # S3, P3 = normaltest(latent3)
    #
    # print(data_type1+': S={0:.3f}   P={1:} '.format(S1[0], P1[0]))  # 返回统计量s和p值
    # print(data_type2+':    S={0:.3f}   P={1:} '.format(S2[0], P2[0]))
    # print(data_type3+':            S={0:.3f}   P={1:} '.format(S3[0], P3[0]))
    # # ######################## 分析数据的偏度和峰度 #################################################
    s1 = pd.Series(latent1.reshape(-1, 1).flatten())
    s2 = pd.Series(latent2.reshape(-1, 1).flatten())
    s3 = pd.Series(latent3.reshape(-1, 1).flatten())

    print(type1+': skew={0:.3f}    kurt={1:.3f} '.format(s1.skew(), s1.kurt()))  # 返回偏度和峰度
    print(type2+':    skew={0:.3f}    kurt={1:.3f} '.format(s2.skew(), s2.kurt()))  #
    print(type3+':            skew={0:.3f}    kurt={1:.3f} '.format(s3.skew(), s3.kurt()))  #
    # # ######################### 分析数据的似然 #################################################
    # l1 = gmm.score(latent1)  # 输入(n_samples, n_features), 输出(n_samples,)
    # l2 = gmm.score(latent2)
    # l3 = gmm.score(latent3)

    # print(type1+': l={0:.3f}'.format(l1))  # 返回似然值
    # print(type2+':    l={0:.3f}'.format(l2))  #
    # print(type3+':            l={0:.3f}'.format(l3))  #

    # # ######################### 分析得分的假阳率 #################################################
    # all_val1_loss = np.load('clustering' + type1 + '/''/epoch/' + str(dict_density_ratio[type1][ratio1]) + '/all_val1_loss' + ratio1 + '.npy', allow_pickle=True)
    # all_test1_loss = np.load('clustering' + type1 + '/''/epoch/' + str(dict_density_ratio[type1][ratio1]) + '/all_test1_loss' + ratio1 + '.npy',  allow_pickle=True)
    # all_val2_loss = np.load('clustering' + type2 + '/''/epoch/' + str(dict_density_AE[type2][ratio1]) + '/all_val1_loss' + ratio1 + '.npy', allow_pickle=True)
    # all_test2_loss = np.load('clustering' + type2 + '/''/epoch/' + str(dict_density_AE[type2][ratio1]) + '/all_test1_loss' + ratio1 + '.npy',  allow_pickle=True)
    # all_val3_loss = np.load('clustering' + type3 + '/''/epoch/' + str(dict_only[type3][ratio1]) + '/all_val1_loss' + ratio1 + '.npy', allow_pickle=True)
    # all_test3_loss = np.load('clustering' + type3 + '/''/epoch/' + str(dict_only[type3][ratio1]) + '/all_test1_loss' + ratio1 + '.npy',  allow_pickle=True)
    #
    # # ############## average ################
    # frame_len = 28
    # all_val1_ave = all_val1_loss
    # all_test1_ave = all_test1_loss
    # all_val2_ave = all_val2_loss
    # all_test2_ave = all_test2_loss
    # all_val3_ave = all_val3_loss
    # all_test3_ave = all_test3_loss
    #
    # all_val1_ave = np.array([sum(x) for x in chunked(all_val1_ave, frame_len)])
    # all_test1_ave = np.array([sum(x) for x in chunked(all_test1_ave, frame_len)])
    # all_val2_ave = np.array([sum(x) for x in chunked(all_val2_ave, frame_len)])
    # all_test2_ave = np.array([sum(x) for x in chunked(all_test2_ave, frame_len)])
    # all_val3_ave = np.array([sum(x) for x in chunked(all_val3_ave, frame_len)])
    # all_test3_ave = np.array([sum(x) for x in chunked(all_test3_ave, frame_len)])
    # print(type1)
    # print('阴性：mean/std: {0:.3f}/{1:.3f}'.format(all_val1_ave.mean(), all_val1_ave.std()))
    # print('阳性：mean/std: {0:.3f}/{1:.3f}'.format(all_test1_ave.mean(), all_test1_ave.std()))
    # print(type2)
    # print('阴性：mean/std: {0:.3f}/{1:.3f}'.format(all_val2_ave.mean(), all_val2_ave.std()))
    # print('阳性：mean/std: {0:.3f}/{1:.3f}'.format(all_test2_ave.mean(), all_test2_ave.std()))
    # print(type3)
    # print('阴性：mean/std: {0:.3f}/{1:.3f}'.format(all_val3_ave.mean(), all_val3_ave.std()))
    # print('阳性：mean/std: {0:.3f}/{1:.3f}'.format(all_test3_ave.mean(), all_test3_ave.std()))


for i in range(9):  # 类型数量
    for j in range(5):  # 比例数量
        alys(data_type1[i], data_type2[i], data_type3[i], ratio[j])

print('完美')















