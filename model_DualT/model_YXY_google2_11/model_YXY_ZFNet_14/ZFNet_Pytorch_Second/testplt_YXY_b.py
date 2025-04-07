# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

plt.figure(1) # 创建图表1
plt.figure(2) # 创建图表2
ax1 = plt.subplot(211) # 在图表2中创建子图1
ax2 = plt.subplot(212) # 在图表2中创建子图2

x = np.linspace(0, 3, 100)
for i in range(5):
    plt.figure(1)  # 选择图表1
    plt.plot(x, np.exp(i*x/3))
    plt.sca(ax1)   # 选择图表2的子图1
    plt.plot(x, np.sin(i*x))
    plt.sca(ax2)  # 选择图表2的子图2
    plt.plot(x, np.cos(i*x))

plt.show()
plt.savefig('testplt_yxy_b.png')
# ————————————————
# 版权声明：本文为CSDN博主「帝壹」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/ydyang1126/article/details/77247654