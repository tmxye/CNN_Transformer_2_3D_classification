#  出现问题UserWarning: Matplotlib is currently using agg, which is a non-GUI backend,
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
# plt.switch_backend('agg')
import matplotlib
from matplotlib import pyplot
matplotlib.use('TkAgg')


# # https://blog.csdn.net/baidu_38963740/article/details/123839178
# # plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# # plt.rcParams['font.serif']=['SimSun'] #Time New Roman
# # https://blog.csdn.net/SL_World/article/details/105390455
# plt.rcParams['font.family']=['Euclid'] #用来正常显示中文标签
# plt.rcParams['font.sans-serif'] = ['SimSun']
# # plt.rcParams['font.serif']=['Euclid Regular'] #用来正常显示中文标签
# # https://www.freesion.com/article/23781280870/
# plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

# https://blog.csdn.net/realitysong/article/details/126711268
# E:\other\yxy\Lib\site-packages\matplotlib\mpl-data\fonts\ttf
# E:\other\yxy\Lib\site-packages\matplotlib\mpl-data\matplotlibrc
plt.rcParams['font.family'] = ['SimSun']        # 2024年改动的检测版本
# plt.rcParams['font.family'] = ['SongEuclid']        # 2023年最终
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


from sklearn.metrics import roc_curve, auc, precision_recall_curve
import pandas as pd
import numpy as np

import csv
# exampleFile = open('./ThreeD/0AlexNet曲线需要的参数.csv')  # 打开csv文件
# exampleFileA = open('./ThreeD/17VGG曲线需要的参数_70.csv')  # 打开csv文件
# exampleFileB = open('./ThreeD/234ResNet曲线需要的参数_245.csv')  # 打开csv文件
# exampleFileC = open('./ThreeD/4023den曲线需要的参数_149.csv')  # 打开csv文件
# exampleFileD = open('./ThreeD/236ResNext曲线需要的参数_165.csv')  # 打开csv文件
# exampleFileE = open('./ThreeD/L109Res2Net曲线需要的参数_154.csv')  # 打开csv文件
# exampleFileF = open('./ThreeD/943seresnet曲线需要的参数.csv')  # 打开csv文件
# exampleFileG = open('./ThreeD/947seresnext曲线需要的参数_247.csv')  # 打开csv文件
# exampleFileH = open('./ThreeD/80Efficinet曲线需要的参数_106.csv')  # 打开csv文件
# exampleFile1 = open('./ThreeD/84Eff曲线需要的参数_144.csv')  # 打开csv文件
# exampleFileA1 = open('./ThreeD/87Efficinet曲线需要的参数_245.csv')  # 打开csv文件
# exampleFileB1 = open('./ThreeD/965RegNetx002曲线需要的参数_249.csv')  # 打开csv文件
# exampleFileC1 = open('./ThreeD/9801reg曲线需要的参数_233.csv')  # 打开csv文件
# exampleFileD1 = open('./ThreeD/65Mobile曲线需要的参数_49.csv')  # 打开csv文件
# exampleFileE1 = open('./ThreeD/L101EFFNet曲线需要的参数_235.csv')  # 打开csv文件
# exampleFileF1 = open('./ThreeD/951CondenseNet曲线需要的参数_62.csv')  # 打开csv文件
# exampleFileG1 = open('./ThreeD/L102ghostnet曲线需要的参数_237.csv')  # 打开csv文件
# # exampleFileG1 = open('./ThreeD/L104_Swin_Con曲线需要的参数_128.csv')  # 打开csv文件
# exampleFileH1 = open('./ThreeD/L106ReLKNet曲线需要的参数_220.csv')  # 打开csv文件
# # exampleFileH1 = open('./ThreeD/L107_Cox_Con曲线需要的参数_213.csv')  # 打开csv文件
# # exampleFileH1 = open('./ThreeD/L107_NAT_Con曲线需要的参数_213.csv')  # 打开csv文件
# # exampleFileH1 = open('./ThreeD/4740-pre107曲线需要的参数_71.csv')  # 打开csv文件


# exampleFile = open('./ThreeD/17VGG曲线需要的参数_70.csv')  # 打开csv文件
exampleFile = open('./ThreeD/10NextVit-S.csv')  # 打开csv文件
exampleFileA = open('./ThreeD/013D-ResNet50.csv')  # 打开csv文件
# exampleFileA = open('./ThreeD/A-3D CNN.csv')  # 打开csv文件
# exampleFileA = open('./ThreeD/234曲线需要的参数_200.csv')  # 打开csv文件
exampleFileB = open('./ThreeD/023D-EfficientNet-b3.csv')  # 打开csv文件
# exampleFileC = open('./ThreeD/033D-ConvNeXt-S.csv')  # 打开csv文件
exampleFileC = open('./ThreeD/033D-EdgeNeXt-S.csv')  # 打开csv文件
exampleFileD = open('./ThreeD/043D-SwinTransformer-S.csv')  # 打开csv文件
exampleFileE = open('./ThreeD/951CondenseNet曲线需要的参数_62.csv')  # 打开csv文件
exampleFileF = open('./ThreeD/063D-BEVT.csv')  # 打开csv文件
exampleFileG = open('./ThreeD/083D-CVT-13.csv')  # 打开csv文件
exampleFileH = open('./ThreeD/951CondenseNet曲线需要的参数_62.csv')  # 打开csv文件
# exampleFile1 = open('./ThreeD/4740_pre107曲线需要的参数_71.csv')  # 打开csv文件
exampleFile1 = open('./ThreeD/113D-MSConvFormer.csv')  # 打开csv文件
# exampleFileA1 = open('./ThreeD/4740_pre107曲线需要的参数_71.csv')  # 打开csv文件
exampleFileA1 = open('./ThreeD/951CondenseNet曲线需要的参数_62.csv')  # 打开csv文件
# exampleFileB1 = open('./ThreeD/965RegNetx002曲线需要的参数_249.csv')  # 打开csv文件
# exampleFileB1 = open('./ThreeD/073D-CoaT-S.csv')  # 打开csv文件
exampleFileB1 = open('./ThreeD/073DMobile-ViT-S.csv')  # 打开csv文件
exampleFileC1 = open('./ThreeD/1005_Conformer.csv')  # 打开csv文件
exampleFileD1 = open('./ThreeD/053D-PoolFormer-S24.csv')  # 这个在04后面
exampleFileE1 = open('./ThreeD/GGG-三.csv')  # 打开csv文件
exampleFileF1 = open('./ThreeD/951CondenseNet曲线需要的参数_62.csv')  # 打开csv文件
exampleFileG1 = open('./ThreeD/951CondenseNet曲线需要的参数_62.csv')  # 打开csv文件
# exampleFileG1 = open('./ThreeD/L104_Swin_Con曲线需要的参数_128.csv')  # 打开csv文件
exampleFileH1 = open('./ThreeD/093D-CMT-S.csv')  # 打开csv文件
# exampleFileH1 = open('./ThreeD/L107_Cox_Con曲线需要的参数_213.csv')  # 打开csv文件
# exampleFileH1 = open('./ThreeD/L108_NAT_Con曲线需要的参数_245.csv')  # 打开csv文件
# exampleFileH1 = open('./ThreeD/4740_pre107曲线需要的参数_71.csv')  # 打开csv文件



exampleReader = csv.reader(exampleFile, delimiter=',')  # 读取csv文件
exampleData = list(exampleReader)  # csv数据转换为列表
length_zu = len(exampleData)  # 得到数据行数

exampleReaderA = csv.reader(exampleFileA, delimiter=',')  # 读取csv文件
exampleDataA = list(exampleReaderA)  # csv数据转换为列表
length_zuA = len(exampleDataA)  # 得到数据行数

exampleReaderB = csv.reader(exampleFileB, delimiter=',')  # 读取csv文件
exampleDataB = list(exampleReaderB)  # csv数据转换为列表
length_zuB = len(exampleDataB)  # 得到数据行数

exampleReaderC = csv.reader(exampleFileC, delimiter=',')  # 读取csv文件
exampleDataC = list(exampleReaderC)  # csv数据转换为列表
length_zuC = len(exampleDataC)  # 得到数据行数

exampleReaderD = csv.reader(exampleFileD, delimiter=',')  # 读取csv文件
exampleDataD = list(exampleReaderD)  # csv数据转换为列表
length_zuD = len(exampleDataD)  # 得到数据行数

exampleReaderE = csv.reader(exampleFileE, delimiter=',')  # 读取csv文件
exampleDataE = list(exampleReaderE)  # csv数据转换为列表
length_zuE = len(exampleDataE)  # 得到数据行数

exampleReaderF = csv.reader(exampleFileF, delimiter=',')  # 读取csv文件
exampleDataF = list(exampleReaderF)  # csv数据转换为列表
length_zuF = len(exampleDataF)  # 得到数据行数

exampleReaderG = csv.reader(exampleFileG, delimiter=',')  # 读取csv文件
exampleDataG = list(exampleReaderG)  # csv数据转换为列表
length_zuG = len(exampleDataG)  # 得到数据行数

exampleReaderH = csv.reader(exampleFileH, delimiter=',')  # 读取csv文件
exampleDataH = list(exampleReaderH)  # csv数据转换为列表
length_zuH = len(exampleDataH)  # 得到数据行数


exampleReader1 = csv.reader(exampleFile1, delimiter=',')  # 读取csv文件
exampleData1 = list(exampleReader1)  # csv数据转换为列表
length_zu1 = len(exampleData1)  # 得到数据行数

exampleReaderA1 = csv.reader(exampleFileA1, delimiter=',')  # 读取csv文件
exampleDataA1 = list(exampleReaderA1)  # csv数据转换为列表
length_zuA1 = len(exampleDataA1)  # 得到数据行数

exampleReaderB1 = csv.reader(exampleFileB1, delimiter=',')  # 读取csv文件
exampleDataB1 = list(exampleReaderB1)  # csv数据转换为列表
length_zuB1 = len(exampleDataB1)  # 得到数据行数

exampleReaderC1 = csv.reader(exampleFileC1, delimiter=',')  # 读取csv文件
exampleDataC1 = list(exampleReaderC1)  # csv数据转换为列表
length_zuC1 = len(exampleDataC1)  # 得到数据行数

exampleReaderD1 = csv.reader(exampleFileD1, delimiter=',')  # 读取csv文件
exampleDataD1 = list(exampleReaderD1)  # csv数据转换为列表
length_zuD1 = len(exampleDataD1)  # 得到数据行数

exampleReaderE1 = csv.reader(exampleFileE1, delimiter=',')  # 读取csv文件
exampleDataE1 = list(exampleReaderE1)  # csv数据转换为列表
length_zuE1 = len(exampleDataE1)  # 得到数据行数

exampleReaderF1 = csv.reader(exampleFileF1, delimiter=',')  # 读取csv文件
exampleDataF1 = list(exampleReaderF1)  # csv数据转换为列表
length_zuF1 = len(exampleDataF1)  # 得到数据行数

exampleReaderG1 = csv.reader(exampleFileG1, delimiter=',')  # 读取csv文件
exampleDataG1 = list(exampleReaderG1)  # csv数据转换为列表
length_zuG1 = len(exampleDataG1)  # 得到数据行数

exampleReaderH1 = csv.reader(exampleFileH1, delimiter=',')  # 读取csv文件
exampleDataH1 = list(exampleReaderH1)  # csv数据转换为列表
length_zuH1 = len(exampleDataH1)  # 得到数据行数


x =list()
y =list()
xA =list()
yA =list()
xB =list()
yB =list()
xC =list()
yC =list()
xD =list()
yD =list()
xE =list()
yE =list()
xF =list()
yF =list()
xG =list()
yG =list()
xH =list()
yH =list()

x1 =list()
y1 =list()
xA1 =list()
yA1 =list()
xB1 =list()
yB1 =list()
xC1 =list()
yC1 =list()
xD1 =list()
yD1 =list()
xE1 =list()
yE1 =list()
xF1 =list()
yF1 =list()
xG1 =list()
yG1 =list()
xH1 =list()
yH1 =list()


for i in range(1, length_zu):  # 从第二行开始读取
    x.append(int(exampleData[i][0]))  # 将第一列数据从第二行读取到最后一行赋给列表x
    y.append(int(exampleData[i][1]))  # 将第二列数据从第二行读取到最后一行赋给列表
output_list1 = np.hstack(x[:])
target_list1 = np.hstack(y[:])

for i in range(1, length_zuA):  # 从第二行开始读取
    xA.append(int(exampleDataA[i][0]))  # 将第一列数据从第二行读取到最后一行赋给列表x
    yA.append(int(exampleDataA[i][1]))  # 将第二列数据从第二行读取到最后一行赋给列表
output_list1A = np.hstack(xA[:])
target_list1A = np.hstack(yA[:])

for i in range(1, length_zuB):  # 从第二行开始读取
    xB.append(int(exampleDataB[i][0]))  # 将第一列数据从第二行读取到最后一行赋给列表x
    yB.append(int(exampleDataB[i][1]))  # 将第二列数据从第二行读取到最后一行赋给列表
output_list1B = np.hstack(xB[:])
target_list1B = np.hstack(yB[:])

for i in range(1, length_zuC):  # 从第二行开始读取
    xC.append(int(exampleDataC[i][0]))  # 将第一列数据从第二行读取到最后一行赋给列表x
    yC.append(int(exampleDataC[i][1]))  # 将第二列数据从第二行读取到最后一行赋给列表
output_list1C = np.hstack(xC[:])
target_list1C = np.hstack(yC[:])

for i in range(1, length_zuD):  # 从第二行开始读取
    xD.append(int(exampleDataD[i][0]))  # 将第一列数据从第二行读取到最后一行赋给列表x
    yD.append(int(exampleDataD[i][1]))  # 将第二列数据从第二行读取到最后一行赋给列表
output_list1D = np.hstack(xD[:])
target_list1D = np.hstack(yD[:])

for i in range(1, length_zuE):  # 从第二行开始读取
    xE.append(int(exampleDataE[i][0]))  # 将第一列数据从第二行读取到最后一行赋给列表x
    yE.append(int(exampleDataE[i][1]))  # 将第二列数据从第二行读取到最后一行赋给列表
output_list1E = np.hstack(xE[:])
target_list1E = np.hstack(yE[:])

for i in range(1, length_zuF):  # 从第二行开始读取
    xF.append(int(exampleDataF[i][0]))  # 将第一列数据从第二行读取到最后一行赋给列表x
    yF.append(int(exampleDataF[i][1]))  # 将第二列数据从第二行读取到最后一行赋给列表
output_list1F = np.hstack(xF[:])
target_list1F = np.hstack(yF[:])

for i in range(1, length_zuG):  # 从第二行开始读取
    xG.append(int(exampleDataG[i][0]))  # 将第一列数据从第二行读取到最后一行赋给列表x
    yG.append(int(exampleDataG[i][1]))  # 将第二列数据从第二行读取到最后一行赋给列表
output_list1G = np.hstack(xG[:])
target_list1G = np.hstack(yG[:])

for i in range(1, length_zuH):  # 从第二行开始读取
    xH.append(int(exampleDataH[i][0]))  # 将第一列数据从第二行读取到最后一行赋给列表x
    yH.append(int(exampleDataH[i][1]))  # 将第二列数据从第二行读取到最后一行赋给列表
output_list1H = np.hstack(xH[:])
target_list1H = np.hstack(yH[:])



for i in range(1, length_zu1):  # 从第二行开始读取
    x1.append(int(exampleData1[i][0]))  # 将第一列数据从第二行读取到最后一行赋给列表x
    y1.append(int(exampleData1[i][1]))  # 将第二列数据从第二行读取到最后一行赋给列表
output_list11 = np.hstack(x1[:])
target_list11 = np.hstack(y1[:])

for i in range(1, length_zuA1):  # 从第二行开始读取
    xA1.append(int(exampleDataA1[i][0]))  # 将第一列数据从第二行读取到最后一行赋给列表x
    yA1.append(int(exampleDataA1[i][1]))  # 将第二列数据从第二行读取到最后一行赋给列表
output_list1A1 = np.hstack(xA1[:])
target_list1A1 = np.hstack(yA1[:])

for i in range(1, length_zuB1):  # 从第二行开始读取
    xB1.append(int(exampleDataB1[i][0]))  # 将第一列数据从第二行读取到最后一行赋给列表x
    yB1.append(int(exampleDataB1[i][1]))  # 将第二列数据从第二行读取到最后一行赋给列表
output_list1B1 = np.hstack(xB1[:])
target_list1B1 = np.hstack(yB1[:])

for i in range(1, length_zuC1):  # 从第二行开始读取
    xC1.append(int(exampleDataC1[i][0]))  # 将第一列数据从第二行读取到最后一行赋给列表x
    yC1.append(int(exampleDataC1[i][1]))  # 将第二列数据从第二行读取到最后一行赋给列表
output_list1C1 = np.hstack(xC1[:])
target_list1C1 = np.hstack(yC1[:])

for i in range(1, length_zuD1):  # 从第二行开始读取
    xD1.append(int(exampleDataD1[i][0]))  # 将第一列数据从第二行读取到最后一行赋给列表x
    yD1.append(int(exampleDataD1[i][1]))  # 将第二列数据从第二行读取到最后一行赋给列表
output_list1D1 = np.hstack(xD1[:])
target_list1D1 = np.hstack(yD1[:])

for i in range(1, length_zuE1):  # 从第二行开始读取
    xE1.append(int(exampleDataE1[i][0]))  # 将第一列数据从第二行读取到最后一行赋给列表x
    yE1.append(int(exampleDataE1[i][1]))  # 将第二列数据从第二行读取到最后一行赋给列表
output_list1E1 = np.hstack(xE1[:])
target_list1E1 = np.hstack(yE1[:])

for i in range(1, length_zuF1):  # 从第二行开始读取
    xF1.append(int(exampleDataF1[i][0]))  # 将第一列数据从第二行读取到最后一行赋给列表x
    yF1.append(int(exampleDataF1[i][1]))  # 将第二列数据从第二行读取到最后一行赋给列表
output_list1F1 = np.hstack(xF1[:])
target_list1F1 = np.hstack(yF1[:])

for i in range(1, length_zuG1):  # 从第二行开始读取
    xG1.append(int(exampleDataG1[i][0]))  # 将第一列数据从第二行读取到最后一行赋给列表x
    yG1.append(int(exampleDataG1[i][1]))  # 将第二列数据从第二行读取到最后一行赋给列表
output_list1G1 = np.hstack(xG1[:])
target_list1G1 = np.hstack(yG1[:])

for i in range(1, length_zuH1):  # 从第二行开始读取
    xH1.append(int(exampleDataH1[i][0]))  # 将第一列数据从第二行读取到最后一行赋给列表x
    yH1.append(int(exampleDataH1[i][1]))  # 将第二列数据从第二行读取到最后一行赋给列表
output_list1H1 = np.hstack(xH1[:])
target_list1H1 = np.hstack(yH1[:])



data = pd.DataFrame(index=range(0, len(target_list1)), columns=('probability', 'The true label'))
data['probability'] = output_list1[:]
data['The true label'] = target_list1[:]
TPRandFPR = pd.DataFrame(index=range(len(target_list1)), columns=('TP', 'FP'))
data.sort_values('probability', inplace=True, ascending=False)  # 改了看能不能随机        #g改了以后ROC曲线成反的了
data1 = []
dict_1 = pd.DataFrame(index=range(len(target_list1)), columns=('P', 'R'))

dataA = pd.DataFrame(index=range(0, len(target_list1A)), columns=('probability', 'The true label'))
dataA['probability'] = output_list1A[:]
dataA['The true label'] = target_list1A[:]
TPRandFPRA = pd.DataFrame(index=range(len(target_list1A)), columns=('TP', 'FP'))
dataA.sort_values('probability', inplace=True, ascending=False)  # 改了看能不能随机        #g改了以后ROC曲线成反的了
data1A = []
dict_1A = pd.DataFrame(index=range(len(target_list1A)), columns=('P', 'R'))

dataB = pd.DataFrame(index=range(0, len(target_list1B)), columns=('probability', 'The true label'))
dataB['probability'] = output_list1B[:]
dataB['The true label'] = target_list1B[:]
TPRandFPRB = pd.DataFrame(index=range(len(target_list1B)), columns=('TP', 'FP'))
dataB.sort_values('probability', inplace=True, ascending=False)  # 改了看能不能随机        #g改了以后ROC曲线成反的了
data1B = []
dict_1B = pd.DataFrame(index=range(len(target_list1B)), columns=('P', 'R'))

dataC = pd.DataFrame(index=range(0, len(target_list1C)), columns=('probability', 'The true label'))
dataC['probability'] = output_list1C[:]
dataC['The true label'] = target_list1C[:]
TPRandFPRC = pd.DataFrame(index=range(len(target_list1C)), columns=('TP', 'FP'))
dataC.sort_values('probability', inplace=True, ascending=False)  # 改了看能不能随机        #g改了以后ROC曲线成反的了
data1C = []
dict_1C = pd.DataFrame(index=range(len(target_list1C)), columns=('P', 'R'))

dataD = pd.DataFrame(index=range(0, len(target_list1D)), columns=('probability', 'The true label'))
dataD['probability'] = output_list1D[:]
dataD['The true label'] = target_list1D[:]
TPRandFPRD = pd.DataFrame(index=range(len(target_list1D)), columns=('TP', 'FP'))
dataD.sort_values('probability', inplace=True, ascending=False)  # 改了看能不能随机        #g改了以后ROC曲线成反的了
data1D = []
dict_1D = pd.DataFrame(index=range(len(target_list1D)), columns=('P', 'R'))

dataE = pd.DataFrame(index=range(0, len(target_list1E)), columns=('probability', 'The true label'))
dataE['probability'] = output_list1E[:]
dataE['The true label'] = target_list1E[:]
TPRandFPRE = pd.DataFrame(index=range(len(target_list1E)), columns=('TP', 'FP'))
dataE.sort_values('probability', inplace=True, ascending=False)  # 改了看能不能随机        #g改了以后ROC曲线成反的了
data1E = []
dict_1E = pd.DataFrame(index=range(len(target_list1E)), columns=('P', 'R'))

dataF = pd.DataFrame(index=range(0, len(target_list1F)), columns=('probability', 'The true label'))
dataF['probability'] = output_list1F[:]
dataF['The true label'] = target_list1F[:]
TPRandFPRF = pd.DataFrame(index=range(len(target_list1F)), columns=('TP', 'FP'))
dataF.sort_values('probability', inplace=True, ascending=False)  # 改了看能不能随机        #g改了以后ROC曲线成反的了
data1F = []
dict_1F = pd.DataFrame(index=range(len(target_list1F)), columns=('P', 'R'))


dataG = pd.DataFrame(index=range(0, len(target_list1G)), columns=('probability', 'The true label'))
dataG['probability'] = output_list1G[:]
dataG['The true label'] = target_list1G[:]
TPRandFPRG = pd.DataFrame(index=range(len(target_list1G)), columns=('TP', 'FP'))
dataG.sort_values('probability', inplace=True, ascending=False)  # 改了看能不能随机        #g改了以后ROC曲线成反的了
data1G = []
dict_1G = pd.DataFrame(index=range(len(target_list1G)), columns=('P', 'R'))

dataH = pd.DataFrame(index=range(0, len(target_list1H)), columns=('probability', 'The true label'))
dataH['probability'] = output_list1H[:]
dataH['The true label'] = target_list1H[:]
TPRandFPRH = pd.DataFrame(index=range(len(target_list1H)), columns=('TP', 'FP'))
dataH.sort_values('probability', inplace=True, ascending=False)  # 改了看能不能随机        #g改了以后ROC曲线成反的了
data1H = []
dict_1H = pd.DataFrame(index=range(len(target_list1H)), columns=('P', 'R'))



fdata1 = pd.DataFrame(index=range(0, len(target_list11)), columns=('probability', 'The true label'))
fdata1['probability'] = output_list11[:]
fdata1['The true label'] = target_list11[:]
TPRandFPR1 = pd.DataFrame(index=range(len(target_list11)), columns=('TP', 'FP'))
fdata1.sort_values('probability', inplace=True, ascending=False)  # 改了看能不能随机        #g改了以后ROC曲线成反的了
data11 = []
dict_11 = pd.DataFrame(index=range(len(target_list11)), columns=('P', 'R'))

dataA1 = pd.DataFrame(index=range(0, len(target_list1A1)), columns=('probability', 'The true label'))
dataA1['probability'] = output_list1A1[:]
dataA1['The true label'] = target_list1A1[:]
TPRandFPRA1 = pd.DataFrame(index=range(len(target_list1A1)), columns=('TP', 'FP'))
dataA1.sort_values('probability', inplace=True, ascending=False)  # 改了看能不能随机        #g改了以后ROC曲线成反的了
data1A1 = []
dict_1A1 = pd.DataFrame(index=range(len(target_list1A1)), columns=('P', 'R'))

dataB1 = pd.DataFrame(index=range(0, len(target_list1B1)), columns=('probability', 'The true label'))
dataB1['probability'] = output_list1B1[:]
dataB1['The true label'] = target_list1B1[:]
TPRandFPRB1 = pd.DataFrame(index=range(len(target_list1B1)), columns=('TP', 'FP'))
dataB1.sort_values('probability', inplace=True, ascending=False)  # 改了看能不能随机        #g改了以后ROC曲线成反的了
data1B1 = []
dict_1B1 = pd.DataFrame(index=range(len(target_list1B1)), columns=('P', 'R'))

dataC1 = pd.DataFrame(index=range(0, len(target_list1C1)), columns=('probability', 'The true label'))
dataC1['probability'] = output_list1C1[:]
dataC1['The true label'] = target_list1C1[:]
TPRandFPRC1 = pd.DataFrame(index=range(len(target_list1C1)), columns=('TP', 'FP'))
dataC1.sort_values('probability', inplace=True, ascending=False)  # 改了看能不能随机        #g改了以后ROC曲线成反的了
data1C1 = []
dict_1C1 = pd.DataFrame(index=range(len(target_list1C1)), columns=('P', 'R'))

dataD1 = pd.DataFrame(index=range(0, len(target_list1D1)), columns=('probability', 'The true label'))
dataD1['probability'] = output_list1D1[:]
dataD1['The true label'] = target_list1D1[:]
TPRandFPRD1 = pd.DataFrame(index=range(len(target_list1D1)), columns=('TP', 'FP'))
dataD1.sort_values('probability', inplace=True, ascending=False)  # 改了看能不能随机        #g改了以后ROC曲线成反的了
data1D1 = []
dict_1D1 = pd.DataFrame(index=range(len(target_list1D1)), columns=('P', 'R'))

dataE1 = pd.DataFrame(index=range(0, len(target_list1E1)), columns=('probability', 'The true label'))
dataE1['probability'] = output_list1E1[:]
dataE1['The true label'] = target_list1E1[:]
TPRandFPRE1 = pd.DataFrame(index=range(len(target_list1E1)), columns=('TP', 'FP'))
dataE1.sort_values('probability', inplace=True, ascending=False)  # 改了看能不能随机        #g改了以后ROC曲线成反的了
data1E1 = []
dict_1E1 = pd.DataFrame(index=range(len(target_list1E1)), columns=('P', 'R'))

dataF1 = pd.DataFrame(index=range(0, len(target_list1F1)), columns=('probability', 'The true label'))
dataF1['probability'] = output_list1F1[:]
dataF1['The true label'] = target_list1F1[:]
TPRandFPRF1 = pd.DataFrame(index=range(len(target_list1F1)), columns=('TP', 'FP'))
dataF1.sort_values('probability', inplace=True, ascending=False)  # 改了看能不能随机        #g改了以后ROC曲线成反的了
data1F1 = []
dict_1F1 = pd.DataFrame(index=range(len(target_list1F1)), columns=('P', 'R'))


dataG1 = pd.DataFrame(index=range(0, len(target_list1G1)), columns=('probability', 'The true label'))
dataG1['probability'] = output_list1G1[:]
dataG1['The true label'] = target_list1G1[:]
TPRandFPRG1 = pd.DataFrame(index=range(len(target_list1G1)), columns=('TP', 'FP'))
dataG1.sort_values('probability', inplace=True, ascending=False)  # 改了看能不能随机        #g改了以后ROC曲线成反的了
data1G1 = []
dict_1G1 = pd.DataFrame(index=range(len(target_list1G1)), columns=('P', 'R'))

dataH1 = pd.DataFrame(index=range(0, len(target_list1H1)), columns=('probability', 'The true label'))
dataH1['probability'] = output_list1H1[:]
dataH1['The true label'] = target_list1H1[:]
TPRandFPRH1 = pd.DataFrame(index=range(len(target_list1H1)), columns=('TP', 'FP'))
dataH1.sort_values('probability', inplace=True, ascending=False)  # 改了看能不能随机        #g改了以后ROC曲线成反的了
data1H1 = []
dict_1H1 = pd.DataFrame(index=range(len(target_list1H1)), columns=('P', 'R'))


for j in range(len(output_list1)):  # 计算全部概率值下的FPR和TPR
    data1 = data.head(n=j + 1)
    FP = len(data1[(data1['The true label'] == 0) & (data1['probability'] >= data.head(len(data1))['probability'])]) / float(len(data[data['The true label'] == 0]))
    TP = len(data1[(data1['The true label'] == 1) & (data1['probability'] >= data.head(len(data1))['probability'])]) / float(len(data[data['The true label'] == 1]))
    TPRandFPR.iloc[j] = [TP, FP]
    TPRandFPR.iloc[0] = [0, 0]

    P = len(data1[(data1['The true label'] == 1) & (data1['probability'] >= data.head(len(data1))['probability'])]) / float(len(data1[(data1['The true label'] == 0) & (data1['probability'] >= data.head(len(data1))['probability'])]) + len(data1[(data1['The true label'] == 1) & (data1['probability'] >= data.head(len(data1))['probability'])]))
    R = len(data1[(data1['The true label'] == 1) & (data1['probability'] >= data.head(len(data1))['probability'])]) / float(len(data[data['The true label'] == 1]))
    dict_1.iloc[j] = [P, R]
    dict_1.iloc[0] = [1, 0]
    dict_1.iloc[len(output_list1)-1] = [0, 1]

for j in range(len(output_list1A)):  # 计算全部概率值下的FPR和TPR
    data1A = dataA.head(n=j + 1)
    FPA = len(data1A[(data1A['The true label'] == 0) & (data1A['probability'] >= dataA.head(len(data1A))['probability'])]) / float(len(dataA[dataA['The true label'] == 0]))
    TPA = len(data1A[(data1A['The true label'] == 1) & (data1A['probability'] >= dataA.head(len(data1A))['probability'])]) / float(len(dataA[dataA['The true label'] == 1]))
    TPRandFPRA.iloc[j] = [TPA, FPA]
    TPRandFPRA.iloc[0] = [0, 0]

    PA = len(data1A[(data1A['The true label'] == 1) & (data1A['probability'] >= dataA.head(len(data1A))['probability'])]) / float(len(data1A[(data1A['The true label'] == 0) & (data1A['probability'] >= dataA.head(len(data1A))['probability'])]) + len(data1A[(data1A['The true label'] == 1) & (data1A['probability'] >= dataA.head(len(data1A))['probability'])]))
    RA = len(data1A[(data1A['The true label'] == 1) & (data1A['probability'] >= dataA.head(len(data1A))['probability'])]) / float(len(dataA[dataA['The true label'] == 1]))
    dict_1A.iloc[j] = [PA, RA]
    dict_1A.iloc[0] = [1, 0]
    dict_1A.iloc[len(output_list1A)-1] = [0, 1]

for j in range(len(output_list1B)):  # 计算全部概率值下的FPR和TPR
    data1B = dataB.head(n=j + 1)
    FPB = len(data1B[(data1B['The true label'] == 0) & (data1B['probability'] >= dataB.head(len(data1B))['probability'])]) / float(len(dataB[dataB['The true label'] == 0]))
    TPB = len(data1B[(data1B['The true label'] == 1) & (data1B['probability'] >= dataB.head(len(data1B))['probability'])]) / float(len(dataB[dataB['The true label'] == 1]))
    TPRandFPRB.iloc[j] = [TPB, FPB]
    TPRandFPRB.iloc[0] = [0, 0]

    PB = len(data1B[(data1B['The true label'] == 1) & (data1B['probability'] >= dataB.head(len(data1B))['probability'])]) / float(len(data1B[(data1B['The true label'] == 0) & (data1B['probability'] >= dataB.head(len(data1B))['probability'])]) + len(data1B[(data1B['The true label'] == 1) & (data1B['probability'] >= dataB.head(len(data1B))['probability'])]))
    RB = len(data1B[(data1B['The true label'] == 1) & (data1B['probability'] >= dataB.head(len(data1B))['probability'])]) / float(len(dataB[dataB['The true label'] == 1]))
    dict_1B.iloc[j] = [PB, RB]
    dict_1B.iloc[0] = [1, 0]
    dict_1B.iloc[len(output_list1B)-1] = [0, 1]

for j in range(len(output_list1C)):  # 计算全部概率值下的FPR和TPR
    data1C = dataC.head(n=j + 1)
    FPC = len(data1C[(data1C['The true label'] == 0) & (data1C['probability'] >= dataC.head(len(data1C))['probability'])]) / float(len(dataC[dataC['The true label'] == 0]))
    TPC = len(data1C[(data1C['The true label'] == 1) & (data1C['probability'] >= dataC.head(len(data1C))['probability'])]) / float(len(dataC[dataC['The true label'] == 1]))
    TPRandFPRC.iloc[j] = [TPC, FPC]
    TPRandFPRC.iloc[0] = [0, 0]

    PC = len(data1C[(data1C['The true label'] == 1) & (data1C['probability'] >= dataC.head(len(data1C))['probability'])]) / float(len(data1C[(data1C['The true label'] == 0) & (data1C['probability'] >= dataC.head(len(data1C))['probability'])]) + len(data1C[(data1C['The true label'] == 1) & (data1C['probability'] >= dataC.head(len(data1C))['probability'])]))
    RC = len(data1C[(data1C['The true label'] == 1) & (data1C['probability'] >= dataC.head(len(data1C))['probability'])]) / float(len(dataC[dataC['The true label'] == 1]))
    dict_1C.iloc[j] = [PC, RC]
    dict_1C.iloc[0] = [1, 0]
    dict_1C.iloc[len(output_list1C)-1] = [0, 1]

for j in range(len(output_list1D)):  # 计算全部概率值下的FPR和TPR
    data1D = dataD.head(n=j + 1)
    FPD = len(data1D[(data1D['The true label'] == 0) & (data1D['probability'] >= dataD.head(len(data1D))['probability'])]) / float(len(dataD[dataD['The true label'] == 0]))
    TPD = len(data1D[(data1D['The true label'] == 1) & (data1D['probability'] >= dataD.head(len(data1D))['probability'])]) / float(len(dataD[dataD['The true label'] == 1]))
    TPRandFPRD.iloc[j] = [TPD, FPD]
    TPRandFPRD.iloc[0] = [0, 0]

    PD = len(data1D[(data1D['The true label'] == 1) & (data1D['probability'] >= dataD.head(len(data1D))['probability'])]) / float(len(data1D[(data1D['The true label'] == 0) & (data1D['probability'] >= dataD.head(len(data1D))['probability'])]) + len(data1D[(data1D['The true label'] == 1) & (data1D['probability'] >= dataD.head(len(data1D))['probability'])]))
    RD = len(data1D[(data1D['The true label'] == 1) & (data1D['probability'] >= dataD.head(len(data1D))['probability'])]) / float(len(dataD[dataD['The true label'] == 1]))
    dict_1D.iloc[j] = [PD, RD]
    dict_1D.iloc[0] = [1, 0]
    dict_1D.iloc[len(output_list1D)-1] = [0, 1]

for j in range(len(output_list1E)):  # 计算全部概率值下的FPR和TPR
    data1E = dataE.head(n=j + 1)
    FPE = len(data1E[(data1E['The true label'] == 0) & (data1E['probability'] >= dataE.head(len(data1E))['probability'])]) / float(len(dataE[dataE['The true label'] == 0]))
    TPE = len(data1E[(data1E['The true label'] == 1) & (data1E['probability'] >= dataE.head(len(data1E))['probability'])]) / float(len(dataE[dataE['The true label'] == 1]))
    TPRandFPRE.iloc[j] = [TPE, FPE]
    TPRandFPRE.iloc[0] = [0, 0]

    PE = len(data1E[(data1E['The true label'] == 1) & (data1E['probability'] >= dataE.head(len(data1E))['probability'])]) / float(len(data1E[(data1E['The true label'] == 0) & (data1E['probability'] >= dataE.head(len(data1E))['probability'])]) + len(data1E[(data1E['The true label'] == 1) & (data1E['probability'] >= dataE.head(len(data1E))['probability'])]))
    RE = len(data1E[(data1E['The true label'] == 1) & (data1E['probability'] >= dataE.head(len(data1E))['probability'])]) / float(len(dataE[dataE['The true label'] == 1]))
    dict_1E.iloc[j] = [PE, RE]
    dict_1E.iloc[0] = [1, 0]
    dict_1E.iloc[len(output_list1E)-1] = [0, 1]

for j in range(len(output_list1F)):  # 计算全部概率值下的FPR和TPR
    data1F = dataF.head(n=j + 1)
    FPF = len(data1F[(data1F['The true label'] == 0) & (data1F['probability'] >= dataF.head(len(data1F))['probability'])]) / float(len(dataF[dataF['The true label'] == 0]))
    TPF = len(data1F[(data1F['The true label'] == 1) & (data1F['probability'] >= dataF.head(len(data1F))['probability'])]) / float(len(dataF[dataF['The true label'] == 1]))
    TPRandFPRF.iloc[j] = [TPF, FPF]
    TPRandFPRF.iloc[0] = [0, 0]

    PF = len(data1F[(data1F['The true label'] == 1) & (data1F['probability'] >= dataF.head(len(data1F))['probability'])]) / float(len(data1F[(data1F['The true label'] == 0) & (data1F['probability'] >= dataF.head(len(data1F))['probability'])]) + len(data1F[(data1F['The true label'] == 1) & (data1F['probability'] >= dataF.head(len(data1F))['probability'])]))
    RF = len(data1F[(data1F['The true label'] == 1) & (data1F['probability'] >= dataF.head(len(data1F))['probability'])]) / float(len(dataF[dataF['The true label'] == 1]))
    dict_1F.iloc[j] = [PF, RF]
    dict_1F.iloc[0] = [1, 0]
    dict_1F.iloc[len(output_list1F)-1] = [0, 1]

for j in range(len(output_list1G)):  # 计算全部概率值下的FPR和TPR
    data1G = dataG.head(n=j + 1)
    FPG = len(data1G[(data1G['The true label'] == 0) & (data1G['probability'] >= dataG.head(len(data1G))['probability'])]) / float(len(dataG[dataG['The true label'] == 0]))
    TPG = len(data1G[(data1G['The true label'] == 1) & (data1G['probability'] >= dataG.head(len(data1G))['probability'])]) / float(len(dataG[dataG['The true label'] == 1]))
    TPRandFPRG.iloc[j] = [TPG, FPG]
    TPRandFPRG.iloc[0] = [0, 0]

    PG = len(data1G[(data1G['The true label'] == 1) & (data1G['probability'] >= dataG.head(len(data1G))['probability'])]) / float(len(data1G[(data1G['The true label'] == 0) & (data1G['probability'] >= dataG.head(len(data1G))['probability'])]) + len(data1G[(data1G['The true label'] == 1) & (data1G['probability'] >= dataG.head(len(data1G))['probability'])]))
    RG = len(data1G[(data1G['The true label'] == 1) & (data1G['probability'] >= dataG.head(len(data1G))['probability'])]) / float(len(dataG[dataG['The true label'] == 1]))
    dict_1G.iloc[j] = [PG, RG]
    dict_1G.iloc[0] = [1, 0]
    dict_1G.iloc[len(output_list1G)-1] = [0, 1]

for j in range(len(output_list1H)):  # 计算全部概率值下的FPR和TPR
    data1H = dataH.head(n=j + 1)
    FPH = len(data1H[(data1H['The true label'] == 0) & (data1H['probability'] >= dataH.head(len(data1H))['probability'])]) / float(len(dataH[dataH['The true label'] == 0]))
    TPH = len(data1H[(data1H['The true label'] == 1) & (data1H['probability'] >= dataH.head(len(data1H))['probability'])]) / float(len(dataH[dataH['The true label'] == 1]))
    TPRandFPRH.iloc[j] = [TPH, FPH]
    TPRandFPRH.iloc[0] = [0, 0]

    PH = len(data1H[(data1H['The true label'] == 1) & (data1H['probability'] >= dataH.head(len(data1H))['probability'])]) / float(len(data1H[(data1H['The true label'] == 0) & (data1H['probability'] >= dataH.head(len(data1H))['probability'])]) + len(data1H[(data1H['The true label'] == 1) & (data1H['probability'] >= dataH.head(len(data1H))['probability'])]))
    RH = len(data1H[(data1H['The true label'] == 1) & (data1H['probability'] >= dataH.head(len(data1H))['probability'])]) / float(len(dataH[dataH['The true label'] == 1]))
    dict_1H.iloc[j] = [PH, RH]
    dict_1H.iloc[0] = [1, 0]
    dict_1H.iloc[len(output_list1H)-1] = [0, 1]




for j in range(len(output_list11)):  # 计算全部概率值下的FPR和TPR
    data11 = fdata1.head(n=j + 1)
    FP = len(data11[(data11['The true label'] == 0) & (data11['probability'] >= fdata1.head(len(data11))['probability'])]) / float(len(fdata1[fdata1['The true label'] == 0]))
    TP = len(data11[(data11['The true label'] == 1) & (data11['probability'] >= fdata1.head(len(data11))['probability'])]) / float(len(fdata1[fdata1['The true label'] == 1]))
    TPRandFPR1.iloc[j] = [TP, FP]
    TPRandFPR1.iloc[0] = [0, 0]

    P = len(data11[(data11['The true label'] == 1) & (data11['probability'] >= fdata1.head(len(data11))['probability'])]) / float(len(data11[(data11['The true label'] == 0) & (data11['probability'] >= fdata1.head(len(data11))['probability'])]) + len(data11[(data11['The true label'] == 1) & (data11['probability'] >= fdata1.head(len(data11))['probability'])]))
    R = len(data11[(data11['The true label'] == 1) & (data11['probability'] >= fdata1.head(len(data11))['probability'])]) / float(len(fdata1[fdata1['The true label'] == 1]))
    dict_11.iloc[j] = [P, R]
    dict_11.iloc[0] = [1, 0]
    dict_11.iloc[len(output_list11)-1] = [0, 1]

for j in range(len(output_list1A1)):  # 计算全部概率值下的FPR和TPR
    data1A = dataA1.head(n=j + 1)
    FPA = len(data1A[(data1A['The true label'] == 0) & (data1A['probability'] >= dataA1.head(len(data1A))['probability'])]) / float(len(dataA1[dataA1['The true label'] == 0]))
    TPA = len(data1A[(data1A['The true label'] == 1) & (data1A['probability'] >= dataA1.head(len(data1A))['probability'])]) / float(len(dataA1[dataA1['The true label'] == 1]))
    TPRandFPRA1.iloc[j] = [TPA, FPA]
    TPRandFPRA1.iloc[0] = [0, 0]

    PA = len(data1A[(data1A['The true label'] == 1) & (data1A['probability'] >= dataA1.head(len(data1A))['probability'])]) / float(len(data1A[(data1A['The true label'] == 0) & (data1A['probability'] >= dataA1.head(len(data1A))['probability'])]) + len(data1A[(data1A['The true label'] == 1) & (data1A['probability'] >= dataA1.head(len(data1A))['probability'])]))
    RA = len(data1A[(data1A['The true label'] == 1) & (data1A['probability'] >= dataA1.head(len(data1A))['probability'])]) / float(len(dataA1[dataA1['The true label'] == 1]))
    dict_1A1.iloc[j] = [PA, RA]
    dict_1A1.iloc[0] = [1, 0]
    dict_1A1.iloc[len(output_list1A1)-1] = [0, 1]

for j in range(len(output_list1B1)):  # 计算全部概率值下的FPR和TPR
    data1B = dataB1.head(n=j + 1)
    FPB = len(data1B[(data1B['The true label'] == 0) & (data1B['probability'] >= dataB1.head(len(data1B))['probability'])]) / float(len(dataB1[dataB1['The true label'] == 0]))
    TPB = len(data1B[(data1B['The true label'] == 1) & (data1B['probability'] >= dataB1.head(len(data1B))['probability'])]) / float(len(dataB1[dataB1['The true label'] == 1]))
    TPRandFPRB1.iloc[j] = [TPB, FPB]
    TPRandFPRB1.iloc[0] = [0, 0]

    PB = len(data1B[(data1B['The true label'] == 1) & (data1B['probability'] >= dataB1.head(len(data1B))['probability'])]) / float(len(data1B[(data1B['The true label'] == 0) & (data1B['probability'] >= dataB1.head(len(data1B))['probability'])]) + len(data1B[(data1B['The true label'] == 1) & (data1B['probability'] >= dataB1.head(len(data1B))['probability'])]))
    RB = len(data1B[(data1B['The true label'] == 1) & (data1B['probability'] >= dataB1.head(len(data1B))['probability'])]) / float(len(dataB1[dataB1['The true label'] == 1]))
    dict_1B1.iloc[j] = [PB, RB]
    dict_1B1.iloc[0] = [1, 0]
    dict_1B1.iloc[len(output_list1B1)-1] = [0, 1]

for j in range(len(output_list1C1)):  # 计算全部概率值下的FPR和TPR
    data1C = dataC1.head(n=j + 1)
    FPC = len(data1C[(data1C['The true label'] == 0) & (data1C['probability'] >= dataC1.head(len(data1C))['probability'])]) / float(len(dataC1[dataC1['The true label'] == 0]))
    TPC = len(data1C[(data1C['The true label'] == 1) & (data1C['probability'] >= dataC1.head(len(data1C))['probability'])]) / float(len(dataC1[dataC1['The true label'] == 1]))
    TPRandFPRC1.iloc[j] = [TPC, FPC]
    TPRandFPRC1.iloc[0] = [0, 0]

    PC = len(data1C[(data1C['The true label'] == 1) & (data1C['probability'] >= dataC1.head(len(data1C))['probability'])]) / float(len(data1C[(data1C['The true label'] == 0) & (data1C['probability'] >= dataC1.head(len(data1C))['probability'])]) + len(data1C[(data1C['The true label'] == 1) & (data1C['probability'] >= dataC1.head(len(data1C))['probability'])]))
    RC = len(data1C[(data1C['The true label'] == 1) & (data1C['probability'] >= dataC1.head(len(data1C))['probability'])]) / float(len(dataC1[dataC1['The true label'] == 1]))
    dict_1C1.iloc[j] = [PC, RC]
    dict_1C1.iloc[0] = [1, 0]
    dict_1C1.iloc[len(output_list1C)-1] = [0, 1]

for j in range(len(output_list1D1)):  # 计算全部概率值下的FPR和TPR
    data1D = dataD1.head(n=j + 1)
    FPD = len(data1D[(data1D['The true label'] == 0) & (data1D['probability'] >= dataD1.head(len(data1D))['probability'])]) / float(len(dataD1[dataD1['The true label'] == 0]))
    TPD = len(data1D[(data1D['The true label'] == 1) & (data1D['probability'] >= dataD1.head(len(data1D))['probability'])]) / float(len(dataD1[dataD1['The true label'] == 1]))
    TPRandFPRD1.iloc[j] = [TPD, FPD]
    TPRandFPRD1.iloc[0] = [0, 0]

    PD = len(data1D[(data1D['The true label'] == 1) & (data1D['probability'] >= dataD1.head(len(data1D))['probability'])]) / float(len(data1D[(data1D['The true label'] == 0) & (data1D['probability'] >= dataD1.head(len(data1D))['probability'])]) + len(data1D[(data1D['The true label'] == 1) & (data1D['probability'] >= dataD1.head(len(data1D))['probability'])]))
    RD = len(data1D[(data1D['The true label'] == 1) & (data1D['probability'] >= dataD1.head(len(data1D))['probability'])]) / float(len(dataD1[dataD1['The true label'] == 1]))
    dict_1D1.iloc[j] = [PD, RD]
    dict_1D1.iloc[0] = [1, 0]
    dict_1D1.iloc[len(output_list1D1)-1] = [0, 1]

for j in range(len(output_list1E)):  # 计算全部概率值下的FPR和TPR
    data1E = dataE1.head(n=j + 1)
    FPE = len(data1E[(data1E['The true label'] == 0) & (data1E['probability'] >= dataE1.head(len(data1E))['probability'])]) / float(len(dataE1[dataE1['The true label'] == 0]))
    TPE = len(data1E[(data1E['The true label'] == 1) & (data1E['probability'] >= dataE1.head(len(data1E))['probability'])]) / float(len(dataE1[dataE1['The true label'] == 1]))
    TPRandFPRE1.iloc[j] = [TPE, FPE]
    TPRandFPRE1.iloc[0] = [0, 0]

    PE = len(data1E[(data1E['The true label'] == 1) & (data1E['probability'] >= dataE1.head(len(data1E))['probability'])]) / float(len(data1E[(data1E['The true label'] == 0) & (data1E['probability'] >= dataE1.head(len(data1E))['probability'])]) + len(data1E[(data1E['The true label'] == 1) & (data1E['probability'] >= dataE1.head(len(data1E))['probability'])]))
    RE = len(data1E[(data1E['The true label'] == 1) & (data1E['probability'] >= dataE1.head(len(data1E))['probability'])]) / float(len(dataE1[dataE1['The true label'] == 1]))
    dict_1E1.iloc[j] = [PE, RE]
    dict_1E1.iloc[0] = [1, 0]
    dict_1E1.iloc[len(output_list1E)-1] = [0, 1]

for j in range(len(output_list1F1)):  # 计算全部概率值下的FPR和TPR
    data1F = dataF1.head(n=j + 1)
    FPF = len(data1F[(data1F['The true label'] == 0) & (data1F['probability'] >= dataF1.head(len(data1F))['probability'])]) / float(len(dataF1[dataF1['The true label'] == 0]))
    TPF = len(data1F[(data1F['The true label'] == 1) & (data1F['probability'] >= dataF1.head(len(data1F))['probability'])]) / float(len(dataF1[dataF1['The true label'] == 1]))
    TPRandFPRF1.iloc[j] = [TPF, FPF]
    TPRandFPRF1.iloc[0] = [0, 0]

    PF = len(data1F[(data1F['The true label'] == 1) & (data1F['probability'] >= dataF1.head(len(data1F))['probability'])]) / float(len(data1F[(data1F['The true label'] == 0) & (data1F['probability'] >= dataF1.head(len(data1F))['probability'])]) + len(data1F[(data1F['The true label'] == 1) & (data1F['probability'] >= dataF1.head(len(data1F))['probability'])]))
    RF = len(data1F[(data1F['The true label'] == 1) & (data1F['probability'] >= dataF1.head(len(data1F))['probability'])]) / float(len(dataF1[dataF1['The true label'] == 1]))
    dict_1F1.iloc[j] = [PF, RF]
    dict_1F1.iloc[0] = [1, 0]
    dict_1F1.iloc[len(output_list1F1)-1] = [0, 1]

for j in range(len(output_list1G1)):  # 计算全部概率值下的FPR和TPR
    data1G = dataG1.head(n=j + 1)
    FPG = len(data1G[(data1G['The true label'] == 0) & (data1G['probability'] >= dataG1.head(len(data1G))['probability'])]) / float(len(dataG1[dataG1['The true label'] == 0]))
    TPG = len(data1G[(data1G['The true label'] == 1) & (data1G['probability'] >= dataG1.head(len(data1G))['probability'])]) / float(len(dataG1[dataG1['The true label'] == 1]))
    TPRandFPRG1.iloc[j] = [TPG, FPG]
    TPRandFPRG1.iloc[0] = [0, 0]

    PG = len(data1G[(data1G['The true label'] == 1) & (data1G['probability'] >= dataG1.head(len(data1G))['probability'])]) / float(len(data1G[(data1G['The true label'] == 0) & (data1G['probability'] >= dataG1.head(len(data1G))['probability'])]) + len(data1G[(data1G['The true label'] == 1) & (data1G['probability'] >= dataG1.head(len(data1G))['probability'])]))
    RG = len(data1G[(data1G['The true label'] == 1) & (data1G['probability'] >= dataG1.head(len(data1G))['probability'])]) / float(len(dataG1[dataG1['The true label'] == 1]))
    dict_1G1.iloc[j] = [PG, RG]
    dict_1G1.iloc[0] = [1, 0]
    dict_1G1.iloc[len(output_list1G)-1] = [0, 1]

for j in range(len(output_list1H1)):  # 计算全部概率值下的FPR和TPR
    data1H = dataH1.head(n=j + 1)
    FPH = len(data1H[(data1H['The true label'] == 0) & (data1H['probability'] >= dataH1.head(len(data1H))['probability'])]) / float(len(dataH1[dataH1['The true label'] == 0]))
    TPH = len(data1H[(data1H['The true label'] == 1) & (data1H['probability'] >= dataH1.head(len(data1H))['probability'])]) / float(len(dataH1[dataH1['The true label'] == 1]))
    TPRandFPRH1.iloc[j] = [TPH, FPH]
    TPRandFPRH1.iloc[0] = [0, 0]

    PH = len(data1H[(data1H['The true label'] == 1) & (data1H['probability'] >= dataH1.head(len(data1H))['probability'])]) / float(len(data1H[(data1H['The true label'] == 0) & (data1H['probability'] >= dataH1.head(len(data1H))['probability'])]) + len(data1H[(data1H['The true label'] == 1) & (data1H['probability'] >= dataH1.head(len(data1H))['probability'])]))
    RH = len(data1H[(data1H['The true label'] == 1) & (data1H['probability'] >= dataH1.head(len(data1H))['probability'])]) / float(len(dataH1[dataH1['The true label'] == 1]))
    dict_1H1.iloc[j] = [PH, RH]
    dict_1H1.iloc[0] = [1, 0]
    dict_1H1.iloc[len(output_list1H1)-1] = [0, 1]



plt.figure(figsize=(5, 3.8), dpi=600)  # 图片长宽和清晰度
# # 画出最终的ROC曲线
# https://blog.csdn.net/xy3233/article/details/122243820
palette = pyplot.get_cmap('Set1')
# plt.plot(TPRandFPR['FP'], TPRandFPR['TP'], ls='-', lw=0.6, color=palette(0), marker='o', markersize=0.5, label='VGG19-AUC=0.9086')
# plt.plot(TPRandFPRA['FP'], TPRandFPRA['TP'], ls='-', lw=0.6, color=palette(1), marker='^', markersize=0.5, label='ResNet101-AUC=0.9443')
# plt.plot(TPRandFPRB['FP'], TPRandFPRB['TP'], ls='-', lw=0.6, color=palette(2), marker='s', markersize=0.5, label='DenseNet121-AUC=0.9457')
# plt.plot(TPRandFPRC['FP'], TPRandFPRC['TP'], ls='-', lw=0.6, color=palette(3), marker='4', markersize=0.5, label='EfficientNetb4-AUC=0.9581')
# plt.plot(TPRandFPRD['FP'], TPRandFPRD['TP'], ls='-', lw=0.6, color=palette(4), marker='2', markersize=0.5, label='RegNetx032-AUC=0.9597')
# plt.plot(TPRandFPRE['FP'], TPRandFPRE['TP'], ls='-', lw=0.6, color=palette(5), marker='*', markersize=0.5, label='CondenseNet-AUC=0.9426')
# plt.plot(TPRandFPRF['FP'], TPRandFPRF['TP'], ls='-', lw=0.6, color=palette(6), marker='1', markersize=0.5, label='GhostNet-AUC=0.9519')
# plt.plot(TPRandFPRG['FP'], TPRandFPRG['TP'], ls='-', lw=0.6, color=palette(7), marker='s', markersize=0.5, label='SwinTransformer-AUC=0.9628')
# plt.plot(TPRandFPRH['FP'], TPRandFPRH['TP'], ls='-', lw=0.6, color=palette(8), marker='3', markersize=0.5, label='NAT-small-AUC=0.9701')
# plt.plot(TPRandFPR1['FP'], TPRandFPR1['TP'], "r--", lw=0.6, marker='o', markersize=0.5, label='RTDenseNet121(Ours)-AUC=0.9783')
# # plt.plot(TPRandFPRA1['FP'], TPRandFPRA1['TP'], ls='-', lw=0.6, color=palette(1), marker='^', markersize=0.5, label='ResNet101-AUC=0.9223')
# # plt.plot(TPRandFPRB1['FP'], TPRandFPRB1['TP'], ls='-', lw=0.6, color=palette(2), marker='s', markersize=0.5, label='DenseNet-AUC=0.9273')
# # plt.plot(TPRandFPRC1['FP'], TPRandFPRC1['TP'], ls='-', lw=0.6, color=palette(3), marker='4', markersize=0.5, label='ResNext101-AUC=0.9313')
# # plt.plot(TPRandFPRD1['FP'], TPRandFPRD1['TP'], ls='-', lw=0.6, color=palette(4), marker='2', markersize=0.5, label='SeResNet101-AUC=0.9205')
# # plt.plot(TPRandFPRE1['FP'], TPRandFPRE1['TP'], ls='-', lw=0.6, color=palette(5), marker='*', markersize=0.5, label='SeResNext101-AUC=0.9360')
# # plt.plot(TPRandFPRF1['FP'], TPRandFPRF1['TP'], ls='-', lw=0.6, color=palette(6), marker='1', markersize=0.5, label='EfficientNetb0-AUC=0.9246')
# # plt.plot(TPRandFPRG1['FP'], TPRandFPRG1['TP'], ls='-', lw=0.6, color=palette(7), marker='s', markersize=0.5, label='EfficientNetb4-AUC=0.9328')
# # plt.plot(TPRandFPRH1['FP'], TPRandFPRH1['TP'], ls='-', lw=0.6, color=palette(0), marker='3', markersize=0.5, label='TARDenseNet121(Ours)-AUC=0.9562')
# # plt.plot(TPRandFPRC['FP'], TPRandFPRC['TP'], ls='-', lw=0.6, color=palette(3), marker='4', markersize=0.5, label='Google')
# # exampleFile = open('./ROC_Chest_2_Cross/VGGroc曲线需要的参数.csv')  # 打开csv文件
# # exampleFileA = open('./ROC_Chest_2_Cross/Res101roc曲线需要的参数.csv')  # 打开csv文件
# # exampleFileB = open('./ROC_Chest_2_Cross/Denroc曲线需要的参数.csv')  # 打开csv文件
# # exampleFileC = open('./ROC_Chest_2_Cross/ResN101roc曲线需要的参数.csv')  # 打开csv文件
# # exampleFileD = open('./ROC_Chest_2_Cross/SERes101roc曲线需要的参数.csv')  # 打开csv文件
# # exampleFileE = open('./ROC_Chest_2_Cross/SEResN101roc曲线需要的参数.csv')  # 打开csv文件
# # exampleFileF = open('./ROC_Chest_2_Cross/Effb0roc曲线需要的参数.csv')  # 打开csv文件
# # exampleFileG = open('./ROC_Chest_2_Cross/Effb4roc曲线需要的参数.csv')  # 打开csv文件
# # exampleFileH = open('./ROC_Chest_2_Cross/JADenroc曲线需要的参数.csv')  # 打开csv文件

# plt.plot(TPRandFPR['FP'], TPRandFPR['TP'], ls='-', lw=0.6, color=palette(8), label='VGG19-AUC=0.9086')
plt.plot(TPRandFPRA['FP'], TPRandFPRA['TP'], ls='-', lw=0.6, color=palette(7), label='3D-ResNet50-AUC=79.47%')
plt.plot(TPRandFPRB['FP'], TPRandFPRB['TP'], ls='-', lw=0.6, color=palette(6), label='3D-EfficientNet-b3-AUC=80.53%')
# plt.plot(TPRandFPRC['FP'], TPRandFPRC['TP'], ls='-', lw=0.6, color=palette(5), label='3D-ConvNeXt-S-AUC=0.8371')
plt.plot(TPRandFPRD['FP'], TPRandFPRD['TP'], ls='-', lw=0.6, color=palette(4), label='3D-SwinTransformer-S-AUC=83.09%')
# plt.plot(TPRandFPRE['FP'], TPRandFPRE['TP'], ls='-', lw=0.6, color=palette(3), label='CondenseNet-AUC=0.9426')
plt.plot(TPRandFPRD1['FP'], TPRandFPRD1['TP'], ls='-', lw=0.6, color=palette(3), label='3D-PoolFormer-S24-AUC=95.43%')
plt.plot(TPRandFPRF['FP'], TPRandFPRF['TP'], ls='-', lw=0.6, color=palette(2), label='3D-BEVT-AUC=95.62%')
plt.plot(TPRandFPRB1['FP'], TPRandFPRB1['TP'], ls='-.', lw=0.6, color=palette(7), label='3D-Mobile-ViT-AUC=83.47%')
plt.plot(TPRandFPRC['FP'], TPRandFPRC['TP'], ls='-', lw=0.6, color=palette(5), label='3D-EdgeNeXt-S-AUC=84.39%')
plt.plot(TPRandFPRG['FP'], TPRandFPRG['TP'], ls='-', lw=0.6, color=palette(1), label='3D-CVT-13-AUC=84.01%')
# plt.plot(TPRandFPRH['FP'], TPRandFPRH['TP'], ls='-', lw=0.6, color=palette(0), label='NAT-small-AUC=0.9701')
plt.plot(TPRandFPRH1['FP'], TPRandFPRH1['TP'], ls='-', lw=0.6, color=palette(8), label='3D-CMT-S-AUC=85.38%')
plt.plot(TPRandFPR['FP'], TPRandFPR['TP'], "b-", lw=0.6, label='3D-NextVit-S-AUC=86.20%')
# plt.plot(TPRandFPRC1['FP'], TPRandFPRB1['TP'], ls='-.', lw=0.6, color=palette(6), label='Conformer-B-AUC=0.9721')
plt.plot(TPRandFPR1['FP'], TPRandFPR1['TP'], "r-", lw=0.6, label='Light-3Dformer(Ours)-AUC=89.81%')

plt.legend()
plt.title('ROC曲线', fontsize=12)
plt.xlim([-0.01, 1.03])
plt.ylim([-0.01, 1.03])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
plt.ylabel('真正类率', fontsize=12)
plt.xlabel('假正类率', fontsize=12)
# plt.show()
print('开始画图啦')
# plt.legend(loc='lower right')  # loc='lower right': 指定示例在右下方
# plt.figure(figsize=(10, 4), dpi=80)  # 图片长宽和清晰度
plt.savefig('./02ROC曲线绘制/21ROC曲线绘制.jpg', dpi=600)
print('画图结束啦')
plt.clf()



# https://www.jianshu.com/p/ef48fff5dd04
precision, recall, thresholds = precision_recall_curve(target_list1, output_list1)
pr_auc = auc(recall, precision)
precisionA, recallA, thresholdsA = precision_recall_curve(target_list1A, output_list1A)
pr_aucA = auc(recallA, precisionA)
precisionB, recallB, thresholdsB = precision_recall_curve(target_list1B, output_list1B)
pr_aucB = auc(recallB, precisionB)
precisionC, recallC, thresholdsC = precision_recall_curve(target_list1C, output_list1C)
pr_aucC = auc(recallC, precisionC)
precisionD, recallD, thresholdsD = precision_recall_curve(target_list1D, output_list1D)
pr_aucD = auc(recallD, precisionD)
precisionE, recallE, thresholdsE = precision_recall_curve(target_list1E, output_list1E)
pr_aucE = auc(recallE, precisionE)
precisionF, recallF, thresholdsF = precision_recall_curve(target_list1F, output_list1F)
pr_aucF = auc(recallF, precisionF)
precisionG, recallG, thresholdsG = precision_recall_curve(target_list1G, output_list1G)
pr_aucG = auc(recallG, precisionG)
precisionH, recallH, thresholdsH = precision_recall_curve(target_list1H, output_list1H)
pr_aucH = auc(recallH, precisionH)

# https://www.jianshu.com/p/ef48fff5dd04
Aprecision, Arecall, Athresholds = precision_recall_curve(target_list11, output_list11)
pr_auc1 = auc(Arecall, Aprecision)
AprecisionA, ArecallA, AthresholdsA = precision_recall_curve(target_list1A1, output_list1A1)
pr_aucA1 = auc(ArecallA, AprecisionA)
AprecisionB, ArecallB, AthresholdsB = precision_recall_curve(target_list1B1, output_list1B1)
pr_aucB1 = auc(ArecallB, AprecisionB)
AprecisionC, ArecallC, AthresholdsC = precision_recall_curve(target_list1C1, output_list1C1)
pr_aucC1 = auc(ArecallC, AprecisionC)
AprecisionD, ArecallD, AthresholdsD = precision_recall_curve(target_list1D1, output_list1D1)
pr_aucD1 = auc(ArecallD, AprecisionD)
AprecisionE, ArecallE, AthresholdsE = precision_recall_curve(target_list1E1, output_list1E1)
pr_aucE1 = auc(ArecallE, AprecisionE)
AprecisionF, ArecallF, AthresholdsF = precision_recall_curve(target_list1F1, output_list1F1)
pr_aucF1 = auc(ArecallF, AprecisionF)
AprecisionG, ArecallG, AthresholdsG = precision_recall_curve(target_list1G1, output_list1G1)
pr_aucG1 = auc(ArecallG, AprecisionG)
AprecisionH, ArecallH, AthresholdsH = precision_recall_curve(target_list1H1, output_list1H1)
pr_aucH1 = auc(ArecallH, AprecisionH)


plt.figure(dpi=600)  # 图片长宽和清晰度
paletteB = pyplot.get_cmap('Set1')
#
# plt.plot(recall, precision, ls='-', lw=0.6, color=paletteB(8), marker='o', markersize=0.5, label='VGG-PR={}'.format(round(pr_auc, 4)))
# plt.plot(recallA, precisionA, ls='-', lw=0.6, color=paletteB(1), marker='^', markersize=0.5, label='ResNet101-PR={}'.format(round(pr_aucA, 4)))
# plt.plot(recallB, precisionB, ls='-', lw=0.6, color=paletteB(2), marker='s', markersize=0.5, label='DenseNet-PR={}'.format(round(pr_aucB, 4)))
# plt.plot(recallC, precisionC, ls='-', lw=0.6, color=paletteB(3), marker='4', markersize=0.5, label='ResNext101-PR={}'.format(round(pr_aucC, 4)))
# plt.plot(recallD, precisionD, ls='-', lw=0.6, color=paletteB(4), marker='2', markersize=0.5, label='SeResNet101-PR={}'.format(round(pr_aucD, 4)))
# plt.plot(recallE, precisionE, ls='-', lw=0.6, color=paletteB(5), marker='*', markersize=0.5, label='SeResNext101-PR={}'.format(round(pr_aucE, 4)))
# plt.plot(recallF, precisionF, ls='-', lw=0.6, color=paletteB(6), marker='1', markersize=0.5, label='EfficientNetb0-PR={}'.format(round(pr_aucF, 4)))
# plt.plot(recallG, precisionG, ls='-', lw=0.6, color=paletteB(7), marker='s', markersize=0.5, label='EfficientNetb4-PR={}'.format(round(pr_aucG, 4)))
# plt.plot(recallH, precisionH, ls='-', lw=0.6, color=paletteB(0), marker='3', markersize=0.5, label='TARDenseNet121(Ours)-PR={}'.format(round(pr_aucH, 4)))
# plt.plot(Arecall, Aprecision, ls='-', lw=0.6, color=paletteB(8), marker='o', markersize=0.5, label='VGG-PR={}'.format(round(pr_auc1, 4)))
# plt.plot(ArecallA, AprecisionA, ls='-', lw=0.6, color=paletteB(1), marker='^', markersize=0.5, label='ResNet101-PR={}'.format(round(pr_aucA1, 4)))
# plt.plot(ArecallB, AprecisionB, ls='-', lw=0.6, color=paletteB(2), marker='s', markersize=0.5, label='DenseNet-PR={}'.format(round(pr_aucB1, 4)))
# plt.plot(ArecallC, AprecisionC, ls='-', lw=0.6, color=paletteB(3), marker='4', markersize=0.5, label='ResNext101-PR={}'.format(round(pr_aucC1, 4)))
# plt.plot(ArecallD, AprecisionD, ls='-', lw=0.6, color=paletteB(4), marker='2', markersize=0.5, label='SeResNet101-PR={}'.format(round(pr_aucD1, 4)))
# plt.plot(ArecallE, AprecisionE, ls='-', lw=0.6, color=paletteB(5), marker='*', markersize=0.5, label='SeResNext101-PR={}'.format(round(pr_aucE1, 4)))
# plt.plot(ArecallF, AprecisionF, ls='-', lw=0.6, color=paletteB(6), marker='1', markersize=0.5, label='EfficientNetb0-PR={}'.format(round(pr_aucF1, 4)))
# plt.plot(ArecallG, AprecisionG, ls='-', lw=0.6, color=paletteB(7), marker='s', markersize=0.5, label='EfficientNetb4-PR={}'.format(round(pr_aucG1, 4)))


# https://blog.csdn.net/xy3233/article/details/122243820?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~aggregatepage~first_rank_ecpm_v1~rank_v31_ecpm-4-122243820-null-null.pc_agg_new_rank&utm_term=pyplot+画两条线&spm=1000.2123.3001.4430


print("round(pr_aucH, 4)", round(pr_aucH, 4))
# plt.plot(recall, precision, ls='-', lw=0.6, color=paletteB(8), label='VGG19-PR={}'.format(round(pr_auc, 4)))
plt.plot(recallA, precisionA, ls='-', lw=0.6, color=paletteB(7), label='3D-ResNet50-PR={}'.format(round(pr_aucA, 4)))
plt.plot(recallB, precisionB, ls='-', lw=0.6, color=paletteB(6), label='3D-EfficientNet-b3-PR={}'.format(round(pr_aucB, 4)))
# plt.plot(recallC, precisionC, ls='-', lw=0.6, color=paletteB(5), label='3D-ConvNeXt-S-PR={}'.format(round(pr_aucC, 4)))
plt.plot(recallC, precisionC, ls='-', lw=0.6, color=paletteB(5), label='3D-EdgeNeXt-S-PR={}'.format(round(pr_aucC, 4)))
plt.plot(recallD, precisionD, ls='-', lw=0.6, color=paletteB(4), label='3D-SwinTransformer-S-PR={}'.format(round(pr_aucD, 4)))
# plt.plot(recallE, precisionE, ls='-', lw=0.6, color=paletteB(3), label='CondenseNet-PR={}'.format(round(pr_aucE, 4)))
plt.plot(ArecallD, AprecisionD, ls='-', lw=0.6, color=paletteB(3), label='3D-PoolFormer-S24-PR={}'.format(round(pr_aucD1, 4)))
plt.plot(recallF, precisionF, ls='-', lw=0.6, color=paletteB(2), label='3D-BEVT-PR={}'.format(round(pr_aucF, 4)))
plt.plot(ArecallB, AprecisionB, ls='-.', lw=0.6, color=paletteB(7), label='3D-Mobile-ViT-PR={}'.format(round(pr_aucB1, 4)))
plt.plot(recallG, precisionG, ls='-', lw=0.6, color=paletteB(1), label='3D-CVT-13-PR={}'.format(round(pr_aucG, 4)))
plt.plot(ArecallH, AprecisionH, ls='-', lw=0.6, color=paletteB(8), label='3D-CMT-S-PR={}'.format(round(pr_aucH, 4)))
plt.plot(recall, precision, "b-", lw=0.6, label='3D-NextVit-S-PR={}'.format(round(pr_auc, 4)))
# plt.plot(ArecallC, AprecisionC, ls='-.', lw=0.6, color=paletteB(6), label='Conformer-B-PR={}'.format(round(pr_aucC1, 4)))
plt.plot(Arecall, Aprecision, "r-", lw=0.6, label='Light-3Dformer(Ours)-PR={}'.format(round(pr_auc1, 4)))


plt.legend()
plt.title('Precision/Recall曲线')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig('./01P_R曲线绘制/02P_R_点曲线绘制D.jpg', dpi=600)
# plt.legend(loc="upper right", labels = ['Full model {}'.format(round(pr_auc, 2)), 'Full model {}'.format(round(pr_auc, 2))])
print('画图一个点的P_R结束啦')
plt.clf()
# plt.show()


# , alpha=0.3

plt.figure(figsize=(5, 3.8), dpi=600)  # 图片长宽和清晰度
# plt.plot(dict_1['R'], dict_1['P'], color='darkorange', label='PR')
paletteA = pyplot.get_cmap('Set1')
# plt.plot(dict_1['R'], dict_1['P'], ls='-', lw=0.6, color=paletteA(8), marker='o', markersize=0.5, label='VGG-PR={}'.format(round(pr_auc, 4)))
# plt.plot(dict_1A['R'], dict_1A['P'], ls='-', lw=0.6, color=paletteA(1), marker='^', markersize=0.5, label='ResNet101-PR={}'.format(round(pr_aucA, 4)))
# plt.plot(dict_1B['R'], dict_1B['P'], ls='-', lw=0.6, color=paletteA(2), marker='s', markersize=0.5, label='DenseNet-PR={}'.format(round(pr_aucB, 4)))
# plt.plot(dict_1C['R'], dict_1C['P'], ls='-', lw=0.6, color=paletteA(3), marker='4', markersize=0.5, label='ResNext101-PR={}'.format(round(pr_aucC, 4)))
# plt.plot(dict_1D['R'], dict_1D['P'], ls='-', lw=0.6, color=paletteA(4), marker='2', markersize=0.5, label='SeResNet101-PR={}'.format(round(pr_aucD, 4)))
# plt.plot(dict_1E['R'], dict_1E['P'], ls='-', lw=0.6, color=paletteA(5), marker='*', markersize=0.5, label='SeResNext101-PR={}'.format(round(pr_aucE, 4)))
# plt.plot(dict_1F['R'], dict_1F['P'], ls='-', lw=0.6, color=paletteA(6), marker='1', markersize=0.5, label='EfficientNetb0-PR={}'.format(round(pr_aucF, 4)))
# plt.plot(dict_1G['R'], dict_1G['P'], ls='-', lw=0.6, color=paletteA(7), marker='s', markersize=0.5, label='EfficientNetb4-PR={}'.format(round(pr_aucG, 4)))
# plt.plot(dict_1H['R'], dict_1H['P'], ls='-', lw=0.6, color=paletteA(0), marker='3', markersize=0.5, label='TARDenseNet121(Ours)-PR={}'.format(round(pr_aucH, 4)))
# plt.plot(dict_11['R'], dict_11['P'], ls='-', lw=0.6, color=paletteA(8), marker='o', markersize=0.5, label='VGG-PR={}'.format(round(pr_auc1, 4)))
# plt.plot(dict_1A['R'], dict_1A['P'], ls='-', lw=0.6, color=paletteA(1), marker='^', markersize=0.5, label='ResNet101-PR={}'.format(round(pr_aucA1, 4)))
# plt.plot(dict_1B['R'], dict_1B['P'], ls='-', lw=0.6, color=paletteA(2), marker='s', markersize=0.5, label='DenseNet-PR={}'.format(round(pr_aucB1, 4)))
# plt.plot(dict_1C['R'], dict_1C['P'], ls='-', lw=0.6, color=paletteA(3), marker='4', markersize=0.5, label='ResNext101-PR={}'.format(round(pr_aucC1, 4)))
# plt.plot(dict_1D['R'], dict_1D['P'], ls='-', lw=0.6, color=paletteA(4), marker='2', markersize=0.5, label='SeResNet101-PR={}'.format(round(pr_aucD1, 4)))
# plt.plot(dict_1E['R'], dict_1E['P'], ls='-', lw=0.6, color=paletteA(5), marker='*', markersize=0.5, label='SeResNext101-PR={}'.format(round(pr_aucE1, 4)))
# plt.plot(dict_1F['R'], dict_1F['P'], ls='-', lw=0.6, color=paletteA(6), marker='1', markersize=0.5, label='EfficientNetb0-PR={}'.format(round(pr_aucF1, 4)))
# plt.plot(dict_1G['R'], dict_1G['P'], ls='-', lw=0.6, color=paletteA(7), marker='s', markersize=0.5, label='EfficientNetb4-PR={}'.format(round(pr_aucG1, 4)))
# plt.plot(dict_1H['R'], dict_1H['P'], ls='-', lw=0.6, color=paletteA(0), marker='3', markersize=0.5, label='TARDenseNet121(Ours)-PR={}'.format(round(pr_aucH1, 4)))

# # plt.plot(dict_1['R'], dict_1['P'], ls='-', lw=0.6, color=paletteA(8), label='VGG19-PR={}'.format(round(pr_auc, 4)))
# plt.plot(dict_1A['R'], dict_1A['P'], ls='-', lw=0.6, color=paletteA(7), label='3D-ResNet50-PR={}'.format(round(pr_aucA, 4)))
# plt.plot(dict_1B['R'], dict_1B['P'], ls='-', lw=0.6, color=paletteA(6), label='3D-EfficientNet-b3-PR={}'.format(round(pr_aucB, 4)))
# # plt.plot(dict_1C['R'], dict_1C['P'], ls='-', lw=0.6, color=paletteA(5), label='3D-ConvNeXt-S-PR={}'.format(round(pr_aucC, 4)))
# plt.plot(dict_1C['R'], dict_1C['P'], ls='-', lw=0.6, color=paletteA(5), label='3D-EdgeNeXt-S-PR={}'.format(round(pr_aucC, 4)))
# plt.plot(dict_1D['R'], dict_1D['P'], ls='-', lw=0.6, color=paletteA(4), label='3D-SwinTransformer-S-PR={}'.format(round(pr_aucD, 4)))
# # plt.plot(dict_1E['R'], dict_1E['P'], ls='-', lw=0.6, color=paletteA(3), label='CondenseNet-PR={}'.format(round(pr_aucE, 4)))
# plt.plot(dict_1D1['R'], dict_1D1['P'], ls='-', lw=0.6, color=paletteA(3), label='3D-PoolFormer-S24-PR={}'.format(round(pr_aucD1, 4)))
# plt.plot(dict_1F['R'], dict_1F['P'], ls='-', lw=0.6, color=paletteA(2), label='3D-BEVT-PR={}'.format(round(pr_aucF, 4)))
# plt.plot(dict_1B1['R'], dict_1B1['P'], ls='-.', lw=0.6, color=paletteA(2), label='3D-Mobile-ViT-PR={}'.format(round(pr_aucB1, 4)))
# plt.plot(dict_1G['R'], dict_1G['P'], ls='-', lw=0.6, color=paletteA(1), label='3D-CVT-13-PR={}'.format(round(pr_aucG, 4)))
# plt.plot(dict_1['R'], dict_1['P'], ls='-', lw=0.6, color=paletteA(8), label='VGG19-PR={}'.format(round(pr_auc, 4)))
plt.plot(dict_1A['R'], dict_1A['P'], ls='-', lw=0.6, color=paletteA(7), label='3D-ResNet50-PR=80.93%')
plt.plot(dict_1B['R'], dict_1B['P'], ls='-', lw=0.6, color=paletteA(6), label='3D-EfficientNet-b3-PR=82.76%')
# plt.plot(dict_1C['R'], dict_1C['P'], ls='-', lw=0.6, color=paletteA(5), label='3D-ConvNeXt-S-PR={}'.format(round(pr_aucC, 4)))
plt.plot(dict_1C['R'], dict_1C['P'], ls='-', lw=0.6, color=paletteA(5), label='3D-EdgeNeXt-S-PR=85.39%')
plt.plot(dict_1D['R'], dict_1D['P'], ls='-', lw=0.6, color=paletteA(4), label='3D-SwinTransformer-S-PR=83.57%')
# plt.plot(dict_1E['R'], dict_1E['P'], ls='-', lw=0.6, color=paletteA(3), label='CondenseNet-PR={}'.format(round(pr_aucE, 4)))
plt.plot(dict_1D1['R'], dict_1D1['P'], ls='-', lw=0.6, color=paletteA(3), label='3D-PoolFormer-S24-PR=85.66%')
plt.plot(dict_1F['R'], dict_1F['P'], ls='-', lw=0.6, color=paletteA(2), label='3D-BEVT-PR=86.01%')
plt.plot(dict_1B1['R'], dict_1B1['P'], ls='-.', lw=0.6, color=paletteA(2), label='3D-Mobile-ViT-PR=84.15%')
plt.plot(dict_1G['R'], dict_1G['P'], ls='-', lw=0.6, color=paletteA(1), label='3D-CVT-13-PR=84.62%')
plt.plot(dict_1H1['R'], dict_1H1['P'], ls='-', lw=0.6, color=paletteA(8), label='3D-CMT-S-PR=85.73%')
# plt.plot(dict_1['R'], dict_1['P'], "b-", lw=0.6, label='3D-NextVit-S-PR={}'.format(round(pr_auc, 4)))
# # plt.plot(dict_1C1['R'], dict_1C1['P'], ls='-.', lw=0.6, color=paletteA(3), label='Conformer-B-PR={}'.format(round(pr_aucC1, 4)))
# plt.plot(dict_11['R'], dict_11['P'], "r-", lw=0.6, label='Light-3Dformer(Ours)-PR={}'.format(round(pr_auc1, 4)))
plt.plot(dict_1['R'], dict_1['P'], "b-", lw=0.6, label='3D-NextVit-S-PR=86.88%')
# plt.plot(dict_1C1['R'], dict_1C1['P'], ls='-.', lw=0.6, color=paletteA(3), label='Conformer-B-PR={}'.format(round(pr_aucC1, 4)))
plt.plot(dict_11['R'], dict_11['P'], "r-", lw=0.6, label='Light-3Dformer(Ours)-PR=89.22%')

plt.legend()
# plt.xlim([-0.01, 1.03])
# plt.ylim([-0.01, 1.03])
plt.title('Precision/Recall曲线', fontsize=12)
plt.xlabel("Recall", fontsize=12)
plt.ylabel("Persion", fontsize=12)
plt.savefig('./01P_R曲线绘制/01P_R曲线绘制D.jpg', dpi=600)
# plt.show()
print('画图P_R结束啦')
plt.clf()
# 值得注意的是，对于特别不平衡的样本，虽然ROC-AUC可能会很好看，但是PR-AUC多半很一般，甚至很不好，
# 上采样和下采样是非常有必要的，另外不要被ROC_AUC所蒙蔽。




def save_epoch_evaluating_indicator(name, accuracy_score, roc_auc_score, recall_score, f1_score, precision_score):
    with open("./save_evaluating_indicator.txt", "a", encoding="utf-8") as f:
        f.write(str({"*************:": name, "*****************:": name}))
        f.write("\n")
        f.write(str({"accuracy_score:": accuracy_score, "sklearn_auc:": roc_auc_score, "recall_score:": recall_score, "f1_score:": f1_score, "precision_score:": precision_score}))
        f.write("\n\n")

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score
print("accuracy_score:", accuracy_score(target_list1, output_list1))
print("sklearn auc:", roc_auc_score(target_list1, output_list1))
print("recall_score:", recall_score(target_list1, output_list1))
print("f1_score:", f1_score(target_list1, output_list1))
print("precision_score:", precision_score(target_list1, output_list1))
accuracy_score1 = accuracy_score(target_list1, output_list1)
roc_auc_score1 = roc_auc_score(target_list1, output_list1)
recall_score1 = recall_score(target_list1, output_list1)
f1_score1 = f1_score(target_list1, output_list1)
precision_score1 = precision_score(target_list1, output_list1)
name = "没有编号"
save_epoch_evaluating_indicator(name, accuracy_score1, roc_auc_score1, recall_score1, f1_score1, precision_score1)

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score
print("A_accuracy_score:", accuracy_score(target_list1A, output_list1A))
print("A_sklearn auc:", roc_auc_score(target_list1A, output_list1A))
print("A_recall_score:", recall_score(target_list1A, output_list1A))
print("A_f1_score:", f1_score(target_list1A, output_list1A))
print("A_precision_score:", precision_score(target_list1A, output_list1A))
accuracy_score1 = accuracy_score(target_list1A, output_list1A)
roc_auc_score1 = roc_auc_score(target_list1A, output_list1A)
recall_score1 = recall_score(target_list1A, output_list1A)
f1_score1 = f1_score(target_list1A, output_list1A)
precision_score1 = precision_score(target_list1A, output_list1A)
name = "AAAAAA"
save_epoch_evaluating_indicator(name, accuracy_score1, roc_auc_score1, recall_score1, f1_score1, precision_score1)

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score
print("B_accuracy_score:", accuracy_score(target_list1B, output_list1B))
print("B_sklearn auc:", roc_auc_score(target_list1B, output_list1B))
print("B_recall_score:", recall_score(target_list1B, output_list1B))
print("B_f1_score:", f1_score(target_list1B, output_list1B))
print("B_precision_score:", precision_score(target_list1B, output_list1B))
accuracy_score1 = accuracy_score(target_list1B, output_list1B)
roc_auc_score1 = roc_auc_score(target_list1B, output_list1B)
recall_score1 = recall_score(target_list1B, output_list1B)
f1_score1 = f1_score(target_list1B, output_list1B)
precision_score1 = precision_score(target_list1B, output_list1B)
name = "BBBBBB"
save_epoch_evaluating_indicator(name, accuracy_score1, roc_auc_score1, recall_score1, f1_score1, precision_score1)

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score
print("C_accuracy_score:", accuracy_score(target_list1C, output_list1C))
print("C_sklearn auc:", roc_auc_score(target_list1C, output_list1C))
print("C_recall_score:", recall_score(target_list1C, output_list1C))
print("C_f1_score:", f1_score(target_list1C, output_list1C))
print("C_precision_score:", precision_score(target_list1C, output_list1C))
accuracy_score1 = accuracy_score(target_list1C, output_list1C)
roc_auc_score1 = roc_auc_score(target_list1C, output_list1C)
recall_score1 = recall_score(target_list1C, output_list1C)
f1_score1 = f1_score(target_list1C, output_list1C)
precision_score1 = precision_score(target_list1C, output_list1C)
name = "CCCCCC"
save_epoch_evaluating_indicator(name, accuracy_score1, roc_auc_score1, recall_score1, f1_score1, precision_score1)

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score
print("D_accuracy_score:", accuracy_score(target_list1D, output_list1D))
print("D_sklearn auc:", roc_auc_score(target_list1D, output_list1D))
print("D_recall_score:", recall_score(target_list1D, output_list1D))
print("D_f1_score:", f1_score(target_list1D, output_list1D))
print("D_precision_score:", precision_score(target_list1D, output_list1D))
name = "DDDDDD"
accuracy_score1 = accuracy_score(target_list1D, output_list1D)
roc_auc_score1 = roc_auc_score(target_list1D, output_list1D)
recall_score1 = recall_score(target_list1D, output_list1D)
f1_score1 = f1_score(target_list1D, output_list1D)
precision_score1 = precision_score(target_list1D, output_list1D)
save_epoch_evaluating_indicator(name, accuracy_score1, roc_auc_score1, recall_score1, f1_score1, precision_score1)

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score
print("E_accuracy_score:", accuracy_score(target_list1E, output_list1E))
print("E_sklearn auc:", roc_auc_score(target_list1E, output_list1E))
print("E_recall_score:", recall_score(target_list1E, output_list1E))
print("E_f1_score:", f1_score(target_list1E, output_list1E))
print("E_precision_score:", precision_score(target_list1E, output_list1E))
accuracy_score1 = accuracy_score(target_list1E, output_list1E)
roc_auc_score1 = roc_auc_score(target_list1E, output_list1E)
recall_score1 = recall_score(target_list1E, output_list1E)
f1_score1 = f1_score(target_list1E, output_list1E)
precision_score1 = precision_score(target_list1E, output_list1E)
name = "EEEEEEE"
save_epoch_evaluating_indicator(name, accuracy_score1, roc_auc_score1, recall_score1, f1_score1, precision_score1)

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score
print("F_accuracy_score:", accuracy_score(target_list1F, output_list1F))
print("F_sklearn auc:", roc_auc_score(target_list1F, output_list1F))
print("F_recall_score:", recall_score(target_list1F, output_list1F))
print("F_f1_score:", f1_score(target_list1F, output_list1F))
print("F_precision_score:", precision_score(target_list1F, output_list1F))
accuracy_score1 = accuracy_score(target_list1F, output_list1F)
roc_auc_score1 = roc_auc_score(target_list1F, output_list1F)
recall_score1 = recall_score(target_list1F, output_list1F)
f1_score1 = f1_score(target_list1F, output_list1F)
precision_score1 = precision_score(target_list1F, output_list1F)
name = "FFFFFFF"
save_epoch_evaluating_indicator(name, accuracy_score1, roc_auc_score1, recall_score1, f1_score1, precision_score1)

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score
print("G_accuracy_score:", accuracy_score(target_list1G, output_list1G))
print("G_sklearn auc:", roc_auc_score(target_list1G, output_list1G))
print("G_recall_score:", recall_score(target_list1G, output_list1G))
print("G_f1_score:", f1_score(target_list1G, output_list1G))
print("G_precision_score:", precision_score(target_list1G, output_list1G))
accuracy_score1 = accuracy_score(target_list1G, output_list1G)
roc_auc_score1 = roc_auc_score(target_list1G, output_list1G)
recall_score1 = recall_score(target_list1G, output_list1G)
f1_score1 = f1_score(target_list1G, output_list1G)
precision_score1 = precision_score(target_list1G, output_list1G)
name = "GGGGGGG"
save_epoch_evaluating_indicator(name, accuracy_score1, roc_auc_score1, recall_score1, f1_score1, precision_score1)

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score
print("H_accuracy_score:", accuracy_score(target_list1H, output_list1H))
print("H_sklearn auc:", roc_auc_score(target_list1H, output_list1H))
print("H_recall_score:", recall_score(target_list1H, output_list1H))
print("H_f1_score:", f1_score(target_list1H, output_list1H))
print("H_precision_score:", precision_score(target_list1H, output_list1H))
accuracy_score1 = accuracy_score(target_list1H, output_list1H)
roc_auc_score1 = roc_auc_score(target_list1H, output_list1H)
recall_score1 = recall_score(target_list1H, output_list1H)
f1_score1 = f1_score(target_list1H, output_list1H)
precision_score1 = precision_score(target_list1H, output_list1H)
name = "HHHHHHH"
save_epoch_evaluating_indicator(name, accuracy_score1, roc_auc_score1, recall_score1, f1_score1, precision_score1)



from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score
print("accuracy_score:", accuracy_score(target_list11, output_list11))
print("sklearn auc:", roc_auc_score(target_list11, output_list11))
print("recall_score:", recall_score(target_list11, output_list11))
print("f1_score:", f1_score(target_list11, output_list11))
print("precision_score:", precision_score(target_list11, output_list11))
accuracy_score1 = accuracy_score(target_list11, output_list11)
roc_auc_score1 = roc_auc_score(target_list11, output_list11)
recall_score1 = recall_score(target_list11, output_list11)
f1_score1 = f1_score(target_list11, output_list11)
precision_score1 = precision_score(target_list11, output_list11)
name = "没有编号"
save_epoch_evaluating_indicator(name, accuracy_score1, roc_auc_score1, recall_score1, f1_score1, precision_score1)

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score
print("A_accuracy_score:", accuracy_score(target_list1A1, output_list1A1))
print("A_sklearn auc:", roc_auc_score(target_list1A1, output_list1A1))
print("A_recall_score:", recall_score(target_list1A1, output_list1A1))
print("A_f1_score:", f1_score(target_list1A1, output_list1A1))
print("A_precision_score:", precision_score(target_list1A1, output_list1A1))
accuracy_score1 = accuracy_score(target_list1A1, output_list1A1)
roc_auc_score1 = roc_auc_score(target_list1A1, output_list1A1)
recall_score1 = recall_score(target_list1A1, output_list1A1)
f1_score1 = f1_score(target_list1A1, output_list1A1)
precision_score1 = precision_score(target_list1A1, output_list1A1)
name = "AAAAAA111111"
save_epoch_evaluating_indicator(name, accuracy_score1, roc_auc_score1, recall_score1, f1_score1, precision_score1)

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score
print("B_accuracy_score:", accuracy_score(target_list1B1, output_list1B1))
print("B_sklearn auc:", roc_auc_score(target_list1B1, output_list1B1))
print("B_recall_score:", recall_score(target_list1B1, output_list1B1))
print("B_f1_score:", f1_score(target_list1B1, output_list1B1))
print("B_precision_score:", precision_score(target_list1B1, output_list1B1))
accuracy_score1 = accuracy_score(target_list1B1, output_list1B1)
roc_auc_score1 = roc_auc_score(target_list1B1, output_list1B1)
recall_score1 = recall_score(target_list1B1, output_list1B1)
f1_score1 = f1_score(target_list1B1, output_list1B1)
precision_score1 = precision_score(target_list1B1, output_list1B1)
name = "BBBBBB111111"
save_epoch_evaluating_indicator(name, accuracy_score1, roc_auc_score1, recall_score1, f1_score1, precision_score1)

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score
print("C_accuracy_score:", accuracy_score(target_list1C1, output_list1C1))
print("C_sklearn auc:", roc_auc_score(target_list1C1, output_list1C1))
print("C_recall_score:", recall_score(target_list1C1, output_list1C1))
print("C_f1_score:", f1_score(target_list1C1, output_list1C1))
print("C_precision_score:", precision_score(target_list1C1, output_list1C1))
accuracy_score1 = accuracy_score(target_list1C1, output_list1C1)
roc_auc_score1 = roc_auc_score(target_list1C1, output_list1C1)
recall_score1 = recall_score(target_list1C1, output_list1C1)
f1_score1 = f1_score(target_list1C1, output_list1C1)
precision_score1 = precision_score(target_list1C1, output_list1C1)
name = "CCCCCC111111"
save_epoch_evaluating_indicator(name, accuracy_score1, roc_auc_score1, recall_score1, f1_score1, precision_score1)

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score
print("D_accuracy_score:", accuracy_score(target_list1D1, output_list1D1))
print("D_sklearn auc:", roc_auc_score(target_list1D1, output_list1D1))
print("D_recall_score:", recall_score(target_list1D1, output_list1D1))
print("D_f1_score:", f1_score(target_list1D1, output_list1D1))
print("D_precision_score:", precision_score(target_list1D1, output_list1D1))
name = "DDDDDD111111"
accuracy_score1 = accuracy_score(target_list1D1, output_list1D1)
roc_auc_score1 = roc_auc_score(target_list1D1, output_list1D1)
recall_score1 = recall_score(target_list1D1, output_list1D1)
f1_score1 = f1_score(target_list1D1, output_list1D1)
precision_score1 = precision_score(target_list1D1, output_list1D1)
save_epoch_evaluating_indicator(name, accuracy_score1, roc_auc_score1, recall_score1, f1_score1, precision_score1)

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score
print("E_accuracy_score:", accuracy_score(target_list1E1, output_list1E1))
print("E_sklearn auc:", roc_auc_score(target_list1E1, output_list1E1))
print("E_recall_score:", recall_score(target_list1E1, output_list1E1))
print("E_f1_score:", f1_score(target_list1E1, output_list1E1))
print("E_precision_score:", precision_score(target_list1E1, output_list1E1))
accuracy_score1 = accuracy_score(target_list1E1, output_list1E1)
roc_auc_score1 = roc_auc_score(target_list1E1, output_list1E1)
recall_score1 = recall_score(target_list1E1, output_list1E1)
f1_score1 = f1_score(target_list1E1, output_list1E1)
precision_score1 = precision_score(target_list1E1, output_list1E1)
name = "EEEEEEE111111"
save_epoch_evaluating_indicator(name, accuracy_score1, roc_auc_score1, recall_score1, f1_score1, precision_score1)

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score
print("F_accuracy_score:", accuracy_score(target_list1F1, output_list1F1))
print("F_sklearn auc:", roc_auc_score(target_list1F1, output_list1F1))
print("F_recall_score:", recall_score(target_list1F1, output_list1F1))
print("F_f1_score:", f1_score(target_list1F1, output_list1F1))
print("F_precision_score:", precision_score(target_list1F1, output_list1F1))
accuracy_score1 = accuracy_score(target_list1F1, output_list1F1)
roc_auc_score1 = roc_auc_score(target_list1F1, output_list1F1)
recall_score1 = recall_score(target_list1F1, output_list1F1)
f1_score1 = f1_score(target_list1F1, output_list1F1)
precision_score1 = precision_score(target_list1F1, output_list1F1)
name = "FFFFFFF111111"
save_epoch_evaluating_indicator(name, accuracy_score1, roc_auc_score1, recall_score1, f1_score1, precision_score1)

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score
print("G_accuracy_score:", accuracy_score(target_list1G1, output_list1G1))
print("G_sklearn auc:", roc_auc_score(target_list1G1, output_list1G1))
print("G_recall_score:", recall_score(target_list1G1, output_list1G1))
print("G_f1_score:", f1_score(target_list1G1, output_list1G1))
print("G_precision_score:", precision_score(target_list1G1, output_list1G1))
accuracy_score1 = accuracy_score(target_list1G1, output_list1G1)
roc_auc_score1 = roc_auc_score(target_list1G1, output_list1G1)
recall_score1 = recall_score(target_list1G1, output_list1G1)
f1_score1 = f1_score(target_list1G1, output_list1G1)
precision_score1 = precision_score(target_list1G1, output_list1G1)
name = "GGGGGGG111111"
save_epoch_evaluating_indicator(name, accuracy_score1, roc_auc_score1, recall_score1, f1_score1, precision_score1)

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score
print("H_accuracy_score:", accuracy_score(target_list1H1, output_list1H1))
print("H_sklearn auc:", roc_auc_score(target_list1H1, output_list1H1))
print("H_recall_score:", recall_score(target_list1H1, output_list1H1))
print("H_f1_score:", f1_score(target_list1H1, output_list1H1))
print("H_precision_score:", precision_score(target_list1H1, output_list1H1))
accuracy_score1 = accuracy_score(target_list1H1, output_list1H1)
roc_auc_score1 = roc_auc_score(target_list1H1, output_list1H1)
recall_score1 = recall_score(target_list1H1, output_list1H1)
f1_score1 = f1_score(target_list1H1, output_list1H1)
precision_score1 = precision_score(target_list1H1, output_list1H1)
name = "HHHHHHH111111"
save_epoch_evaluating_indicator(name, accuracy_score1, roc_auc_score1, recall_score1, f1_score1, precision_score1)

import itertools


##########      混淆矩阵可视化     #######
fpr, tpr, threshold = roc_curve(output_list1, target_list1)
cm = np.arange(4).reshape(2, 2)
a, b, c, d = 0, 0, 0, 0
for i in range(len(output_list1)):
    if (output_list1[i] == 0) and (target_list1[i] < 0.5):  # TN
        a = a + 1
    cm[0, 0] = a
    if (output_list1[i] == 0) and (target_list1[i] >= 0.5):  # FP
        b = b + 1
    cm[1, 0] = b
    if (output_list1[i] == 1) and (target_list1[i] < 0.5):  # FN
        c = c + 1
    cm[0, 1] = c
    if (output_list1[i] == 1) and (target_list1[i] >= 0.5):  # TP
        d = d + 1
    cm[1, 1] = d
classes = [0, 1]
plt.figure(figsize=(4.6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('NextVit-S', fontsize=30)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=0, fontsize=20)
plt.yticks(tick_marks, classes, fontsize=20)
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=30)
plt.ylabel('True label', fontsize=18)
plt.xlabel('Predicted label', fontsize=18)
plt.savefig('./03混淆矩阵/03混淆矩阵_1.jpg', dpi=600)
plt.clf()


##########      混淆矩阵可视化     #######
fpr, tpr, threshold = roc_curve(output_list1A, target_list1A)
cm = np.arange(4).reshape(2, 2)
a, b, c, d = 0, 0, 0, 0
for i in range(len(output_list1A)):
    if (output_list1A[i] == 0) and (target_list1A[i] < 0.5):  # TN
        a = a + 1
    cm[0, 0] = a
    if (output_list1A[i] == 0) and (target_list1A[i] >= 0.5):  # FP
        b = b + 1
    cm[1, 0] = b
    if (output_list1A[i] == 1) and (target_list1A[i] < 0.5):  # FN
        c = c + 1
    cm[0, 1] = c
    if (output_list1A[i] == 1) and (target_list1A[i] >= 0.5):  # TP
        d = d + 1
    cm[1, 1] = d
classes = [0, 1]
plt.figure(figsize=(4.6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('3D-ResNet50', fontsize=30)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=0, fontsize=20)
plt.yticks(tick_marks, classes, fontsize=20)
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=30)
plt.ylabel('True label', fontsize=18)
plt.xlabel('Predicted label', fontsize=18)
plt.savefig('./03混淆矩阵/03混淆矩阵_A.jpg', dpi=600)
plt.clf()


##########      混淆矩阵可视化     #######
fpr, tpr, threshold = roc_curve(output_list1B, target_list1B)
cm = np.arange(4).reshape(2, 2)
a, b, c, d = 0, 0, 0, 0
for i in range(len(output_list1B)):
    if (output_list1B[i] == 0) and (target_list1B[i] < 0.5):  # TN
        a = a + 1
    cm[0, 0] = a
    if (output_list1B[i] == 0) and (target_list1B[i] >= 0.5):  # FP
        b = b + 1
    cm[1, 0] = b
    if (output_list1B[i] == 1) and (target_list1B[i] < 0.5):  # FN
        c = c + 1
    cm[0, 1] = c
    if (output_list1B[i] == 1) and (target_list1B[i] >= 0.5):  # TP
        d = d + 1
    cm[1, 1] = d
classes = [0, 1]
plt.figure(figsize=(4.6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('3D-EfficientNet-b3', fontsize=30)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=0, fontsize=20)
plt.yticks(tick_marks, classes, fontsize=20)
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=30)
plt.ylabel('True label', fontsize=18)
plt.xlabel('Predicted label', fontsize=18)
plt.savefig('./03混淆矩阵/03混淆矩阵_B.jpg', dpi=600)
plt.clf()


##########      混淆矩阵可视化     #######
fpr, tpr, threshold = roc_curve(output_list1C, target_list1C)
cm = np.arange(4).reshape(2, 2)
a, b, c, d = 0, 0, 0, 0
for i in range(len(output_list1C)):
    if (output_list1C[i] == 0) and (target_list1C[i] < 0.5):  # TN
        a = a + 1
    cm[0, 0] = a
    if (output_list1C[i] == 0) and (target_list1C[i] >= 0.5):  # FP
        b = b + 1
    cm[1, 0] = b
    if (output_list1C[i] == 1) and (target_list1C[i] < 0.5):  # FN
        c = c + 1
    cm[0, 1] = c
    if (output_list1C[i] == 1) and (target_list1C[i] >= 0.5):  # TP
        d = d + 1
    cm[1, 1] = d
classes = [0, 1]
plt.figure(figsize=(4.6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
# plt.title('3D-ConvNeXt-S', fontsize=30)
plt.title('3D-EdgeNeXt-S', fontsize=30)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=0, fontsize=20)
plt.yticks(tick_marks, classes, fontsize=20)
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=30)
plt.ylabel('True label', fontsize=18)
plt.xlabel('Predicted label', fontsize=18)
plt.savefig('./03混淆矩阵/03混淆矩阵_C.jpg', dpi=600)
plt.clf()


##########      混淆矩阵可视化     #######
fpr, tpr, threshold = roc_curve(output_list1D, target_list1D)
cm = np.arange(4).reshape(2, 2)
a, b, c, d = 0, 0, 0, 0
for i in range(len(output_list1D)):
    if (output_list1D[i] == 0) and (target_list1D[i] < 0.5):  # TN
        a = a + 1
    cm[0, 0] = a
    if (output_list1D[i] == 0) and (target_list1D[i] >= 0.5):  # FP
        b = b + 1
    cm[1, 0] = b
    if (output_list1D[i] == 1) and (target_list1D[i] < 0.5):  # FN
        c = c + 1
    cm[0, 1] = c
    if (output_list1D[i] == 1) and (target_list1D[i] >= 0.5):  # TP
        d = d + 1
    cm[1, 1] = d
classes = [0, 1]
plt.figure(figsize=(4.6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('3D-SwinTransformer-S', fontsize=30)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=0, fontsize=20)
plt.yticks(tick_marks, classes, fontsize=20)
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=30)
plt.ylabel('True label', fontsize=18)
plt.xlabel('Predicted label', fontsize=18)
plt.savefig('./03混淆矩阵/03混淆矩阵_D.jpg', dpi=600)
plt.clf()


##########      混淆矩阵可视化     #######
fpr, tpr, threshold = roc_curve(output_list1E, target_list1E)
cm = np.arange(4).reshape(2, 2)
a, b, c, d = 0, 0, 0, 0
for i in range(len(output_list1E)):
    if (output_list1E[i] == 0) and (target_list1E[i] < 0.5):  # TN
        a = a + 1
    cm[0, 0] = a
    if (output_list1E[i] == 0) and (target_list1E[i] >= 0.5):  # FP
        b = b + 1
    cm[1, 0] = b
    if (output_list1E[i] == 1) and (target_list1E[i] < 0.5):  # FN
        c = c + 1
    cm[0, 1] = c
    if (output_list1E[i] == 1) and (target_list1E[i] >= 0.5):  # TP
        d = d + 1
    cm[1, 1] = d
classes = [0, 1]
plt.figure(figsize=(4.6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix', fontsize=30)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=0, fontsize=20)
plt.yticks(tick_marks, classes, fontsize=20)
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=30)
plt.ylabel('True label', fontsize=18)
plt.xlabel('Predicted label', fontsize=18)
plt.savefig('./03混淆矩阵/03混淆矩阵_E.jpg', dpi=600)
plt.clf()


##########      混淆矩阵可视化     #######
fpr, tpr, threshold = roc_curve(output_list1F, target_list1F)
cm = np.arange(4).reshape(2, 2)
a, b, c, d = 0, 0, 0, 0
for i in range(len(output_list1F)):
    if (output_list1F[i] == 0) and (target_list1F[i] < 0.5):  # TN
        a = a + 1
    cm[0, 0] = a
    if (output_list1F[i] == 0) and (target_list1F[i] >= 0.5):  # FP
        b = b + 1
    cm[1, 0] = b
    if (output_list1F[i] == 1) and (target_list1F[i] < 0.5):  # FN
        c = c + 1
    cm[0, 1] = c
    if (output_list1F[i] == 1) and (target_list1F[i] >= 0.5):  # TP
        d = d + 1
    cm[1, 1] = d
classes = [0, 1]
plt.figure(figsize=(4.6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('3D-BEVT', fontsize=30)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=0, fontsize=20)
plt.yticks(tick_marks, classes, fontsize=20)
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=30)
plt.ylabel('True label', fontsize=18)
plt.xlabel('Predicted label', fontsize=18)
plt.savefig('./03混淆矩阵/03混淆矩阵_F.jpg', dpi=600)
plt.clf()

##########      混淆矩阵可视化     #######
fpr, tpr, threshold = roc_curve(output_list1G, target_list1G)
cm = np.arange(4).reshape(2, 2)
a, b, c, d = 0, 0, 0, 0
for i in range(len(output_list1G)):
    if (output_list1G[i] == 0) and (target_list1G[i] < 0.5):  # TN
        a = a + 1
    cm[0, 0] = a
    if (output_list1G[i] == 0) and (target_list1G[i] >= 0.5):  # FP
        b = b + 1
    cm[1, 0] = b
    if (output_list1G[i] == 1) and (target_list1G[i] < 0.5):  # FN
        c = c + 1
    cm[0, 1] = c
    if (output_list1G[i] == 1) and (target_list1G[i] >= 0.5):  # TP
        d = d + 1
    cm[1, 1] = d
classes = [0, 1]
plt.figure(figsize=(4.6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('3D-CVT-13', fontsize=30)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=0, fontsize=20)
plt.yticks(tick_marks, classes, fontsize=20)
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=30)
plt.ylabel('True label', fontsize=18)
plt.xlabel('Predicted label', fontsize=18)
plt.savefig('./03混淆矩阵/03混淆矩阵_G.jpg', dpi=600)
plt.clf()


##########      混淆矩阵可视化     #######
fpr, tpr, threshold = roc_curve(output_list1H, target_list1H)
cm = np.arange(4).reshape(2, 2)
a, b, c, d = 0, 0, 0, 0
for i in range(len(output_list1H)):
    if (output_list1H[i] == 0) and (target_list1H[i] < 0.5):  # TN
        a = a + 1
    cm[0, 0] = a
    if (output_list1H[i] == 0) and (target_list1H[i] >= 0.5):  # FP
        b = b + 1
    cm[1, 0] = b
    if (output_list1H[i] == 1) and (target_list1H[i] < 0.5):  # FN
        c = c + 1
    cm[0, 1] = c
    if (output_list1H[i] == 1) and (target_list1H[i] >= 0.5):  # TP
        d = d + 1
    cm[1, 1] = d
classes = [0, 1]
plt.figure(figsize=(4.6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix', fontsize=30)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=0, fontsize=20)
plt.yticks(tick_marks, classes, fontsize=20)
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=30)
plt.ylabel('True label', fontsize=18)
plt.xlabel('Predicted label', fontsize=18)
plt.savefig('./03混淆矩阵/03混淆矩阵_H.jpg', dpi=600)
plt.clf()


##########      混淆矩阵可视化     #######
fpr, tpr, threshold = roc_curve(output_list11, target_list11)
cm = np.arange(4).reshape(2, 2)
a, b, c, d = 0, 0, 0, 0
for i in range(len(output_list11)):
    if (output_list11[i] == 0) and (target_list11[i] < 0.5):  # TN
        a = a + 1
    cm[0, 0] = a
    if (output_list11[i] == 0) and (target_list11[i] >= 0.5):  # FP
        b = b + 1
    cm[1, 0] = b
    if (output_list11[i] == 1) and (target_list11[i] < 0.5):  # FN
        c = c + 1
    cm[0, 1] = c
    if (output_list11[i] == 1) and (target_list11[i] >= 0.5):  # TP
        d = d + 1
    cm[1, 1] = d
classes = [0, 1]
plt.figure(figsize=(4.6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Light-3Dformer', fontsize=30)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=0, fontsize=20)
plt.yticks(tick_marks, classes, fontsize=20)
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=30)
plt.ylabel('True label', fontsize=18)
plt.xlabel('Predicted label', fontsize=18)
plt.savefig('./03混淆矩阵/03混淆矩阵_A1A.jpg', dpi=600)
plt.clf()

##########      混淆矩阵可视化     #######
fpr, tpr, threshold = roc_curve(output_list1A1, target_list1A1)
cm = np.arange(4).reshape(2, 2)
a, b, c, d = 0, 0, 0, 0
for i in range(len(output_list1A1)):
    if (output_list1A1[i] == 0) and (target_list1A1[i] < 0.5):  # TN
        a = a + 1
    cm[0, 0] = a
    if (output_list1A1[i] == 0) and (target_list1A1[i] >= 0.5):  # FP
        b = b + 1
    cm[1, 0] = b
    if (output_list1A1[i] == 1) and (target_list1A1[i] < 0.5):  # FN
        c = c + 1
    cm[0, 1] = c
    if (output_list1A1[i] == 1) and (target_list1A1[i] >= 0.5):  # TP
        d = d + 1
    cm[1, 1] = d
classes = [0, 1]
plt.figure(figsize=(4.6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix', fontsize=30)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=0, fontsize=20)
plt.yticks(tick_marks, classes, fontsize=20)
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=30)
plt.ylabel('True label', fontsize=18)
plt.xlabel('Predicted label', fontsize=18)
plt.savefig('./03混淆矩阵/03混淆矩阵_A1.jpg', dpi=600)
plt.clf()

##########      混淆矩阵可视化     #######
fpr, tpr, threshold = roc_curve(output_list1B1, target_list1B1)
cm = np.arange(4).reshape(2, 2)
a, b, c, d = 0, 0, 0, 0
for i in range(len(output_list1B1)):
    if (output_list1B1[i] == 0) and (target_list1B1[i] < 0.5):  # TN
        a = a + 1
    cm[0, 0] = a
    if (output_list1B1[i] == 0) and (target_list1B1[i] >= 0.5):  # FP
        b = b + 1
    cm[1, 0] = b
    if (output_list1B1[i] == 1) and (target_list1B1[i] < 0.5):  # FN
        c = c + 1
    cm[0, 1] = c
    if (output_list1B1[i] == 1) and (target_list1B1[i] >= 0.5):  # TP
        d = d + 1
    cm[1, 1] = d
classes = [0, 1]
plt.figure(figsize=(4.6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
# plt.title('3D-CoaT-S', fontsize=30)
plt.title('3D-Mobile-ViT', fontsize=30)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=0, fontsize=20)
plt.yticks(tick_marks, classes, fontsize=20)
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=30)
plt.ylabel('True label', fontsize=18)
plt.xlabel('Predicted label', fontsize=18)
plt.savefig('./03混淆矩阵/03混淆矩阵_B1.jpg', dpi=600)
plt.clf()

##########      混淆矩阵可视化     #######
fpr, tpr, threshold = roc_curve(output_list1C1, target_list1C1)
cm = np.arange(4).reshape(2, 2)
a, b, c, d = 0, 0, 0, 0
for i in range(len(output_list1C)):
    if (output_list1C1[i] == 0) and (target_list1C1[i] < 0.5):  # TN
        a = a + 1
    cm[0, 0] = a
    if (output_list1C1[i] == 0) and (target_list1C1[i] >= 0.5):  # FP
        b = b + 1
    cm[1, 0] = b
    if (output_list1C1[i] == 1) and (target_list1C1[i] < 0.5):  # FN
        c = c + 1
    cm[0, 1] = c
    if (output_list1C1[i] == 1) and (target_list1C1[i] >= 0.5):  # TP
        d = d + 1
    cm[1, 1] = d
classes = [0, 1]
plt.figure(figsize=(4.6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Conformer-B', fontsize=30)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=0, fontsize=20)
plt.yticks(tick_marks, classes, fontsize=20)
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=30)
plt.ylabel('True label', fontsize=18)
plt.xlabel('Predicted label', fontsize=18)
plt.savefig('./03混淆矩阵/03混淆矩阵_C1.jpg', dpi=600)
plt.clf()

##########      混淆矩阵可视化     #######
fpr, tpr, threshold = roc_curve(output_list1D1, target_list1D1)
cm = np.arange(4).reshape(2, 2)
a, b, c, d = 0, 0, 0, 0
for i in range(len(output_list1D)):
    if (output_list1D1[i] == 0) and (target_list1D1[i] < 0.5):  # TN
        a = a + 1
    cm[0, 0] = a
    if (output_list1D1[i] == 0) and (target_list1D1[i] >= 0.5):  # FP
        b = b + 1
    cm[1, 0] = b
    if (output_list1D1[i] == 1) and (target_list1D1[i] < 0.5):  # FN
        c = c + 1
    cm[0, 1] = c
    if (output_list1D1[i] == 1) and (target_list1D1[i] >= 0.5):  # TP
        d = d + 1
    cm[1, 1] = d
classes = [0, 1]
plt.figure(figsize=(4.6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('3D-PoolFormer-S24', fontsize=30)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=0, fontsize=20)
plt.yticks(tick_marks, classes, fontsize=20)
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=30)
plt.ylabel('True label', fontsize=18)
plt.xlabel('Predicted label', fontsize=18)
plt.savefig('./03混淆矩阵/03混淆矩阵_D1.jpg', dpi=600)
plt.clf()

##########      混淆矩阵可视化     #######
fpr, tpr, threshold = roc_curve(output_list1E1, target_list1E1)
cm = np.arange(4).reshape(2, 2)
a, b, c, d = 0, 0, 0, 0
for i in range(len(output_list1E1)):
    if (output_list1E1[i] == 0) and (target_list1E1[i] < 0.5):  # TN
        a = a + 1
    cm[0, 0] = a
    if (output_list1E1[i] == 0) and (target_list1E1[i] >= 0.5):  # FP
        b = b + 1
    cm[1, 0] = b
    if (output_list1E1[i] == 1) and (target_list1E1[i] < 0.5):  # FN
        c = c + 1
    cm[0, 1] = c
    if (output_list1E1[i] == 1) and (target_list1E1[i] >= 0.5):  # TP
        d = d + 1
    cm[1, 1] = d
classes = [0, 1]
plt.figure(figsize=(4.6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix', fontsize=30)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=0, fontsize=20)
plt.yticks(tick_marks, classes, fontsize=20)
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=30)
plt.ylabel('True label', fontsize=18)
plt.xlabel('Predicted label', fontsize=18)
plt.savefig('./03混淆矩阵/03混淆矩阵_E1.jpg', dpi=600)
plt.clf()

##########      混淆矩阵可视化     #######
fpr, tpr, threshold = roc_curve(output_list1F1, target_list1F1)
cm = np.arange(4).reshape(2, 2)
a, b, c, d = 0, 0, 0, 0
for i in range(len(output_list1F1)):
    if (output_list1F1[i] == 0) and (target_list1F1[i] < 0.5):  # TN
        a = a + 1
    cm[0, 0] = a
    if (output_list1F1[i] == 0) and (target_list1F1[i] >= 0.5):  # FP
        b = b + 1
    cm[1, 0] = b
    if (output_list1F1[i] == 1) and (target_list1F1[i] < 0.5):  # FN
        c = c + 1
    cm[0, 1] = c
    if (output_list1F1[i] == 1) and (target_list1F1[i] >= 0.5):  # TP
        d = d + 1
    cm[1, 1] = d
classes = [0, 1]
plt.figure(figsize=(4.6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix', fontsize=30)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=0, fontsize=20)
plt.yticks(tick_marks, classes, fontsize=20)
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=30)
plt.ylabel('True label', fontsize=18)
plt.xlabel('Predicted label', fontsize=18)
plt.savefig('./03混淆矩阵/03混淆矩阵_F1.jpg', dpi=600)
plt.clf()

##########      混淆矩阵可视化     #######
fpr, tpr, threshold = roc_curve(output_list1G1, target_list1G1)
cm = np.arange(4).reshape(2, 2)
a, b, c, d = 0, 0, 0, 0
for i in range(len(output_list1G1)):
    if (output_list1G1[i] == 0) and (target_list1G1[i] < 0.5):  # TN
        a = a + 1
    cm[0, 0] = a
    if (output_list1G1[i] == 0) and (target_list1G1[i] >= 0.5):  # FP
        b = b + 1
    cm[1, 0] = b
    if (output_list1G1[i] == 1) and (target_list1G1[i] < 0.5):  # FN
        c = c + 1
    cm[0, 1] = c
    if (output_list1G1[i] == 1) and (target_list1G1[i] >= 0.5):  # TP
        d = d + 1
    cm[1, 1] = d
classes = [0, 1]
plt.figure(figsize=(4.6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix', fontsize=30)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=0, fontsize=20)
plt.yticks(tick_marks, classes, fontsize=20)
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=30)
plt.ylabel('True label', fontsize=18)
plt.xlabel('Predicted label', fontsize=18)
plt.savefig('./03混淆矩阵/03混淆矩阵_G1.jpg', dpi=600)
plt.clf()

##########      混淆矩阵可视化     #######
fpr, tpr, threshold = roc_curve(output_list1H1, target_list1H1)
cm = np.arange(4).reshape(2, 2)
a, b, c, d = 0, 0, 0, 0
for i in range(len(output_list1H1)):
    if (output_list1H1[i] == 0) and (target_list1H1[i] < 0.5):  # TN
        a = a + 1
    cm[0, 0] = a
    if (output_list1H1[i] == 0) and (target_list1H1[i] >= 0.5):  # FP
        b = b + 1
    cm[1, 0] = b
    if (output_list1H1[i] == 1) and (target_list1H1[i] < 0.5):  # FN
        c = c + 1
    cm[0, 1] = c
    if (output_list1H1[i] == 1) and (target_list1H1[i] >= 0.5):  # TP
        d = d + 1
    cm[1, 1] = d
classes = [0, 1]
plt.figure(figsize=(4.6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('3D-CMT-S', fontsize=30)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=0, fontsize=20)
plt.yticks(tick_marks, classes, fontsize=20)
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=30)
plt.ylabel('True label', fontsize=18)
plt.xlabel('Predicted label', fontsize=18)
plt.savefig('./03混淆矩阵/03混淆矩阵_H1.jpg', dpi=600)
plt.clf()




