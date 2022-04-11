# -*- coding: utf-8 -*-
"""
@Time: 2022/3/25 15:57
@Author: TK <TK@bupt.edu.cn>
@Software: PyCharm
"""

# # 打开文件
# fo = open("runoob.txt", "r")
# print("文件名为: ", fo.name)
#
# line = fo.read(10)
# line2 = fo.read(10)
# print("读取的字符串: %s" % line)
# print("读取的字符串2: %s" % line2)
# # 关闭文件
# fo.close()
import numpy as np
import matplotlib.pyplot as plt

t = 10
sample = np.random.exponential(t, size=10000)  # 产生10000个满足指数分布的随机数
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.hist(sample, bins=80, alpha=0.7, density=True)  # bins也就是总共有几条条状图,density表示条形图的总面积为1.
plt.margins(0.02)

# 根据公式绘制指数分布的概率密度函数
lam = 1 / t
x = np.arange(0, 80, 0.1)
y = lam * np.exp(- lam * x)
plt.plot(x, y, color='orange', lw=3)  # 设置标题和坐标轴
plt.title('指数概率密度函数, 1/λ=10')
plt.xlabel('时间')
plt.ylabel('阻塞率')
plt.show()

