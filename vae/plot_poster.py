# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
#
# # plt.rcParams['font.sans-serif'] = ['SimHei']  # 防止有中文而出现乱码
#
# labels = ['a', 'b', 'd', 'e', 'f']
# AE = [0.816, 0.583, 0.667, 0.922, 0.835]
# VAE_0 = [0.823, 0.579, 0.750, 0.924, 0.827]
# VAE_001 = [0.825, 0.561, 0.607, 0.923, 0.823]
# VAE_01 = [0.821, 0.551, 0.583, 0.918, 0.824]
# VAE_1 = [0.798, 0.559, 0.607, 0.881, 0.824]
# VAE_10 = [0.800, 0.557, 0.631, 0.881, 0.805]
# VAE_100 = [0.803, 0.551, 0.691, 0.881, 0.804]
#
#
# x = np.arange(len(labels))  # the label locations
# width = 0.12  # the width of the bars
#
# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width*3, AE, width, label='AE')
# rects2 = ax.bar(x - width*2, VAE_0, width, label='VAE_0')
# rects3 = ax.bar(x - width*1, VAE_001, width, label='VAE_0.01')
# rects4 = ax.bar(x,           VAE_01, width, label='VAE_0.1')
# rects5 = ax.bar(x + width*1, VAE_1, width, label='VAE_1')
# rects6 = ax.bar(x + width*2, VAE_10, width, label='VAE_10')
# rects7 = ax.bar(x + width*3, VAE_100, width, label='VAE_100')
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('AUC', fontsize=26)
# plt.ylim(0.5, 1)
# ax.set_title('The AUC values of single subset', fontsize=22)
# ax.set_xticks(x)
# ax.set_xticklabels(labels, fontsize=20)
# plt.tick_params(labelsize=20)  # 刻度打大小
# plt.xticks(fontsize=30)  # 横坐标标注大小
# ax.legend(fontsize=16)  # 右上角标注的大小
# plt.rcParams['figure.dpi'] = 3000  # 分辨率
#
#
# def autolabel(rects):
#     """Attach a text label above each bar in *rects*, displaying its height."""
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')
#
#
# autolabel(rects1)
# autolabel(rects2)
# autolabel(rects3)
# autolabel(rects4)
# autolabel(rects5)
# autolabel(rects6)
# autolabel(rects7)
#
# # fig.tight_layout()
#
# plt.show()

# #####################################################################################################################
# #####################################################################################################################
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams['font.sans-serif'] = ['SimHei']  # 防止有中文而出现乱码

labels = ['ae', 'ef', 'all']
AE = [0.820, 0.847, 0.783]
VAE_0 = [0.823, 0.890, 0.782]
VAE_001 = [0.822, 0.899, 0.786]
VAE_01 = [0.810, 0.891, 0.750]
VAE_1 = [0.766, 0.838, 0.678]
VAE_10 = [0.765, 0.836, 0.678]
VAE_100 = [0.765, 0.836, 0.644]

# fig, ax = plt.subplots()
fig, ax = plt.subplots()
x = np.arange(len(labels))  # the label locations
width = 0.12  # the width of the bars


rects1 = ax.bar(x - width*3, AE, width, label='AE')
rects2 = ax.bar(x - width*2, VAE_0, width, label='VAE_0')
rects3 = ax.bar(x - width*1, VAE_001, width, label='VAE_0.01')
rects4 = ax.bar(x,           VAE_01, width, label='VAE_0.1')
rects5 = ax.bar(x + width*1, VAE_1, width, label='VAE_1')
rects6 = ax.bar(x + width*2, VAE_10, width, label='VAE_10')
rects7 = ax.bar(x + width*3, VAE_100, width, label='VAE_100')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('AUC', fontsize=26)
plt.ylim(0.5, 1)
ax.set_title('The AUC values of the three sets of subset', fontsize=22)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=20)
plt.tick_params(labelsize=20)  # 刻度打大小
plt.xticks(fontsize=30)  # 横坐标标注大小
ax.legend(fontsize=16)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)
autolabel(rects6)
autolabel(rects7)

fig.tight_layout()

plt.show()
