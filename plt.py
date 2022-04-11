# -*- coding: utf-8 -*-
"""
@Time: 2022/3/26 21:51
@Author: TK <TK@bupt.edu.cn>
@Software: PyCharm
"""

# 绘图
import numpy as np
from matplotlib import pyplot as plt

lambda_req = 12
bp = np.load('data' + '/' + 'bp_arr_all_' + str(lambda_req) + '.npy')  # 阻塞率
resource = np.load('data' + '/' + 'resource_util_all_' + str(lambda_req) + '.npy')  # 资源利用率


fig = plt.figure()
font0 = {'family': 'Times New Roman', 'weight': 'normal', 'style': 'italic', 'size': 18}
font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}
font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
font3 = {'family': 'Times New Roman', 'weight': 'normal', 'style': 'italic', 'size': 15}
ax1 = fig.add_subplot()
ax2 = ax1.twinx()
lns1 = ax1.plot(bp*100, label='Blocking rate', color='r')
# lns2 = ax1.plot(val1_loss, label='Resource utilization', color='b')
# lns3 = ax1.plot(test1_loss, label='test_e_abnormal', color='y')
# lns4 = ax1.plot(val2_loss, label='test_other_normal', color='purple')
# lns5 = ax1.plot(test2_loss, label='test_other_abnormal', color='pink')

lns6 = ax2.plot(resource*100, label='Resource utilization', color='g')
# lns7 = ax2.plot(auc_other_ave, label='AUC_other_ave', color='cyan')
# lns8 = ax2.plot(auc_ae_amin, label='AUC_e_amin', color='k')
# lns9 = ax2.plot(auc_other_amin, label='AUC_other_amin', color='tan')
# lns10 = ax2.plot(auc_ae_amax, label='AUC_e_amax', color='lime')
# lns11 = ax2.plot(auc_other_amax, label='AUC_other_max', color='coral')

plt.title('lambda_req:' + str(lambda_req), font3)
# lns = lns1 + lns2 + lns3 + lns4 + lns5 + lns6 + lns7 + lns8 + lns9 + lns10 + lns11
lns = lns1 + lns6
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)
ax1.set_xlabel('Epoch', font3)
ax1.set_ylabel('Blocking rate', font3)
ax2.set_ylabel('Resource utilization', font3)
ax1.set_ylim(0, 100)
ax2.set_ylim(0, 100)
plt.savefig('clustering/result_' + str(lambda_req) + '.png')
plt.show()
