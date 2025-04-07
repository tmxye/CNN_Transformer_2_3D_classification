import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import random
from matplotlib.font_manager import FontProperties
from pylab import *

# 每隔两小时range(2， 26， 2) ,数据在x轴的数据， 可迭代
x = range(2, 26, 2)
y = [15, 16, 14, 17, 20, 25, 26, 24, 22, 18, 15, 10]

fig = plt.figure(figsize=(20, 8), dpi=80)  # (20, 8)宽20，高8，dpi设置图片清晰度， 让图片更加清晰
plt.plot(x, y)

# 设置x轴, y轴的刻度
_xticks_labels = [i / 2 for i in range(2, 49)]
plt.xticks(_xticks_labels[::4])  # 设置步长
plt.yticks(range(min(y), max(y) + 1))
plt.title(u'中文')
plt.show()
plt.savefig('testplt_YXY.jpg')
# import cv2
# img = cv2.imread('0029.png')
# print(img)




# 原文链接：https: // blog.csdn.net / weixin_39121325 / article / details / 89310795