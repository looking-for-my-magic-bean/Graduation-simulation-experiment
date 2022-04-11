import numpy as np
from sklearn.manifold import TSNE
from pandas.core.frame import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold

data_type1 = '_d_a_density_AE'  # _only_a  _a_a_density_ratio
ratio1 = '1.00-1.00'  # 0.01-1.00   1.00-1.00   100.00-1.00
data_type2 = '_only_d'  # _only_a  _a_a_density_ratio
ratio2 = '100.00-1.00'  # 0.01-1.00   1.00-1.00   100.00-1.00
latent1 = np.load('clustering' + data_type1 + '/''/epoch/' + str(15) + '/all_training_latent' + ratio1 + '.npy', allow_pickle=True)
latent2 = np.load('clustering' + data_type2 + '/''/epoch/' + str(23) + '/all_training_latent' + ratio2 + '.npy', allow_pickle=True)
color1 = np.array(["r"]*latent1.shape[0])
color2 = np.array(["g"]*latent2.shape[0])
label1 = np.zeros(latent1.shape[0])
label2 = np.ones(latent2.shape[0])

latent = np.concatenate((latent1, latent2), axis=0)
label = np.hstack((label1, label2))

# #####################################

S_X1 = latent  # 原始特征
class_num = 2
# ####生成标签
y_s = label
# for i in range(20 * class_num):  # 让y变成[0,0,0,..,1,1,1,....,9,9]
#     y_s[i] = i // 20
# ##################
# 设置散点形状
maker = ['o', 'v', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']
# 设置散点颜色
colors = ['r', 'g', 'yellow', 'cyan', 'blue', 'lime', 'r', 'violet', 'm', 'peru', 'olivedrab',
          'hotpink']
# 图例名称
Label_Com = ['density_AE', 'β-VAE', 'S-2', 'T-2', 'S-3',
             'T-3', 'S-4', 'T-4', 'S-5', 'T-5', 'S-6', 'T-6', 'S-7', 'T-7', 'S-8', 'T-8', 'S-9', 'T-9',
             'S-10', 'T-10', 'S-11', 'T-11', 'S-12', 'T-12']
# 设置字体格式
font1 = {'family': 'Times New Roman',

         'weight': 'bold',
         'size': 32,
         }


def visual(X):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=999)
    X_tsne = tsne.fit_transform(X)

    print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    # '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    return X_norm


def plot_with_labels(S_lowDWeights, Trure_labels, name):
    plt.cla()  # 清除当前图形中的当前活动轴,所以可以重复利用

    # 降到二维了，分别给x和y
    True_labels = Trure_labels.reshape((-1, 1))

    S_data = np.hstack((S_lowDWeights, True_labels))
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})

    for index in range(class_num):
        X = S_data.loc[S_data['label'] == index]['x']
        Y = S_data.loc[S_data['label'] == index]['y']
        plt.scatter(X, Y, cmap='brg', s=100, marker=maker[0], c=colors[index], alpha=0.65)
        # plt.scatter(X, Y, cmap='brg', s=50, marker=maker[0], c='', edgecolors=colors[index], alpha=0.65)

    plt.xticks([])  # 去掉横坐标值
    plt.yticks([])  # 去掉纵坐标值
    #
    plt.title(name, fontsize=32, fontweight='normal', pad=20)


fig = plt.figure(figsize=(9, 9))
ax1 = fig.add_subplot(111)
plot_with_labels(visual(S_X1), y_s, 'd_'+ratio1)

plt.subplots_adjust(left=None, bottom=0.15, right=None, top=None,
                    wspace=0.1, hspace=0.15)
plt.legend(scatterpoints=1, labels=Label_Com, loc='best', labelspacing=0.4, columnspacing=0.4, markerscale=2,
           bbox_to_anchor=(0.9, 0), ncol=12, prop=font1, handletextpad=0.1)

plt.savefig('./TSNE/new_AE_d_'+ratio1+'.png', format='png', dpi=300, bbox_inches='tight')
plt.show()


