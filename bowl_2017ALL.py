import h5py, os
import matplotlib.pyplot as plt
import numpy as np
# 读取数据
import pandas as pd

# https://www.jb51.net/article/227343.htm
data = pd.read_csv("./data_science_bowl_2017/000JPG/stage1_labels.csv")
# print(data) # 用户编号  性别  年龄(岁)  年收入(元)  是否购买
# (1)获取第1列，并存入一个数组中
col_1 = data["id"]  #获取一列，用一维数据
col_2 = data["cancer"]  #获取一列，用一维数据
# print("len", len(col_2))

import glob
import os
# file_path = r'E:\文件夹1\文件夹2'  # 文件夹位置
# file = glob.glob(os.path.join(file_path, "*.csv"))  # 文件列表
# file.sort()  # 文件列表按名称排序
# data = pd.read_csv(file[i])

import os
import pydicom       #用于读取DICOM(DCOM)文件
import argparse
# import scipy.misc    #用imageio替代
import imageio

import SimpleITK as sitk
import cv2


def convert_from_dicom_to_jpg(img, low_window, high_window, save_path):
    lungwin = np.array([low_window * 1., high_window * 1.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg = (newimg * 255).astype('uint16') # 再转换至0-255，且将编码方式由原来的unit16转换为unit8
    # 用cv2写入图像指令，保存jpg即可
    cv2.imwrite(save_path, newimg, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

def save(epoch):
    with open("./data_science_bowl_2017/000JPG/bowl_2017_ALL.txt", "a", encoding="utf-8") as f:
        f.write(str(epoch))
        f.write("\n")
def save_0(epoch):
    with open("./data_science_bowl_2017/000JPG/bowl_2017_ALL0.txt", "a", encoding="utf-8") as f:
        f.write(str(epoch))
        f.write("\n")

"""
发现图片不是按照顺序的，这里采用的是二维的读取方式，这里准备采用三维读取
https://zhuanlan.zhihu.com/p/339458544
还有一种是dicom包的方式，     https://blog.csdn.net/qq_35054151/article/details/112086843
"""



import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 包含所有患者目录的根目录
INPUT_FOLDER = './data_science_bowl_2017'
patients = os.listdir(INPUT_FOLDER)
patients.sort()

def load_scan(path):
    # print("os.listdir(path)", os.listdir(path))
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    # slices = [print("(path + '/' + s)", (path + '/' + s)) for s in os.listdir(path)]
    """    raw_data_element = next(de_gen)
RuntimeError: generator raised StopIteration
    路径没出现问题，但报错，后来发现是系统的问题：https://www.thinbug.com/q/53296469
    修改的            # raise StopIteration  # at end of file            # by YXY            return
    """
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness
    return slices


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # 转换为int16，int16是ok的，因为所有的数值都应该 <32k
    image = image.astype(np.int16)

    # 设置边界外的元素为0
    image[image == -2000] = 0

    # 转换为HU单位
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


# first_patient_pixels = get_pixels_hu(first_patient)
# plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
# plt.xlabel("Hounsfield Units (HU)")
# plt.ylabel("Frequency")
# plt.show()
#
# # 显示一个中间位置的切片
# plt.imshow(first_patient_pixels[80], cmap=plt.cm.gray)
# plt.show()



# for i in range(1100, len(col_2)):
for i in range(len(col_2)):
    # https://blog.csdn.net/gisaavg/article/details/124479985
    file_path = r'./data_science_bowl_2017/' + str(col_1[i]) + '/'    # 文件夹位置
    threeD_file_path = r'./data_science_bowl_2017/' + str(col_1[i])   # 文件夹位置
    file = glob.glob(os.path.join(file_path, "*.dcm"))  # 文件列表
    # # file.sort()  # 文件列表按名称排序
    # threeD_patient = load_scan(threeD_file_path)
    # threeD_patient_pixels = get_pixels_hu(threeD_patient)
    continueK = 0

    # # 出现错误：         进行跳过
    # if i <= 480:
    #     continue
    # if i <= 1329:
    #     continue


    if col_2[i] == 1:
        # 用python批量把DICOM(dcm)转换成jpg图像  https://blog.csdn.net/you2336/article/details/107615846
        for j in range(len(file)):
            # if j >= 16:
            #     continue
            if continueK == 2:
                continue
            # file.sort()  # 文件列表按名称排序
            threeD_patient = load_scan(threeD_file_path)
            threeD_patient_pixels = get_pixels_hu(threeD_patient)
            print("tumor")
            print("file", file[j])
            filename = file[j][25:][:-4]  # 文件名
            # print("filename", filename)
            # python 截取数据到 斜杆/      https://blog.csdn.net/m0_46400195/article/details/125599965
            list2 = filename.split('\\')  # 采用"."进行分割       \\
            # print("list2[0]", list2[0])
            if os.path.exists('./data_science_bowl_2017/000JPG/001JPGALL/' + list2[0]) and continueK==0:
                continueK = 2
                continue
            if not os.path.exists('./data_science_bowl_2017/000JPG/001JPGALL/' + list2[0]):
                continueK = 1
                os.mkdir('./data_science_bowl_2017/000JPG/001JPGALL/' + list2[0])

            filename = list2[0] + '/' + str(j)            # 这样省去重新命名     只取16个，1-16的命名方式
            # save("./data_science_bowl_2017/000JPG/001JPG/%s.jpg" % filename)
            if j == 0:
                save("./data_science_bowl_2017/000JPG/001JPGALL/%s.jpg" % (list2[0] + '/' + str(0)))

            # ds = pydicom.read_file(file[j])          #读取文件
            # img = ds.pixel_array
            # by YXY
            img = threeD_patient_pixels[j]

            # error = False
            # # print("img", img.shape)
            # # print("imgAA", img)
            # for i in range(512):
            #     for j in range(512):
            #         # if img[i][j] < -2048 or img[i][j] > 2048:
            #         # if img[i][j] < -2049 or img[i][j] > 2049:
            #         if img[i][j] < -28008 or img[i][j] > 25555:
            #             error = True
            #         if error == True:
            #             continue
            #     # j = i
            #     # if img[i][j] < -2049 or img[i][j] > 2049 or img[511-i][511-j] < -2049 or img[511-i][511-j] > 2049 or\
            #     #     img[511-i][j] < -2049 or img[511-i][j] > 2049 or img[i][511 - j] < -2049 or img[i][511 - j] > 2049:
            #     #     error = True
            #     if error == True:
            #         continue
            # if error == True:
            #     continue


            imageio.imwrite("./data_science_bowl_2017/000JPG/001JPGALL/%s.jpg" % filename, img)

            # 修改过后的批量转换且无纯黑图像问题     https://blog.csdn.net/ambrace/article/details/124662370
            ds_array = sitk.ReadImage("./data_science_bowl_2017/000JPG/001JPGALL/%s.jpg" % filename)
            img_array = sitk.GetArrayFromImage(ds_array)
            shape = img_array.shape
            img_array = np.reshape(img_array, (shape[0], shape[1]))  # 获取array中的height和width
            high = np.max(img_array)
            low = np.min(img_array)
            output_jpg_path = "./data_science_bowl_2017/000JPG/001JPGALL/%s.jpg" % filename
            convert_from_dicom_to_jpg(img_array, low, high, output_jpg_path)


    elif col_2[i] == 0:
        # 用python批量把DICOM(dcm)转换成jpg图像  https://blog.csdn.net/you2336/article/details/107615846
        for j in range(len(file)):
            # if j >= 16:
            #     continue
            if continueK == 2:
                continue
            # file.sort()  # 文件列表按名称排序
            threeD_patient = load_scan(threeD_file_path)
            threeD_patient_pixels = get_pixels_hu(threeD_patient)
            print("normal")
            print("file", file[j])
            filename = file[j][25:][:-4]  # 文件名
            # print("filename", filename)
            # python 截取数据到 斜杆/      https://blog.csdn.net/m0_46400195/article/details/125599965
            list2 = filename.split('\\')  # 采用"."进行分割       \\
            # print("list2[0]", list2[0])
            if os.path.exists('./data_science_bowl_2017/000JPG/000JPGALL/' + list2[0]) and continueK==0:
                continueK = 2
                continue
            if not os.path.exists('./data_science_bowl_2017/000JPG/000JPGALL/' + list2[0]):
                continueK = 1
                os.mkdir('./data_science_bowl_2017/000JPG/000JPGALL/' + list2[0])

            filename = list2[0] + '/' + str(j)            # 这样省去重新命名     只取16个，1-16的命名方式
            # save("./data_science_bowl_2017/000JPG/000JPG/%s.jpg" % filename)
            if j == 0:
                save_0("./data_science_bowl_2017/000JPG/000JPGALL/%s.jpg" % (list2[0] + '/' + str(0)))

            # ds = pydicom.read_file(file[j])          #读取文件
            # img = ds.pixel_array
            # by YXY
            img = threeD_patient_pixels[j]

            # # print("img", img.shape)
            # # print("imgAA", img)
            # error = False
            # for i in range(512):
            #     for j in range(512):
            #         # if img[i][j] < -2048 or img[i][j] > 2048:
            #         # if img[i][j] < -2049 or img[i][j] > 2049:
            #         if img[i][j] < -28008 or img[i][j] > 25555:
            #             error = True
            #         if error == True:
            #             continue
            #     # j = i
            #     # if img[i][j] < -2049 or img[i][j] > 2049 or img[511-i][511-j] < -2049 or img[511-i][511-j] > 2049 or\
            #     #     img[511-i][j] < -2049 or img[511-i][j] > 2049 or img[i][511 - j] < -2049 or img[i][511 - j] > 2049:
            #     #     error = True
            #     if error == True:
            #         continue
            # if error == True:
            #     continue


            imageio.imwrite("./data_science_bowl_2017/000JPG/000JPGALL/%s.jpg" % filename, img)

            # 修改过后的批量转换且无纯黑图像问题     https://blog.csdn.net/ambrace/article/details/124662370
            ds_array = sitk.ReadImage("./data_science_bowl_2017/000JPG/000JPGALL/%s.jpg" % filename)
            img_array = sitk.GetArrayFromImage(ds_array)
            shape = img_array.shape
            img_array = np.reshape(img_array, (shape[0], shape[1]))  # 获取array中的height和width
            high = np.max(img_array)
            low = np.min(img_array)
            output_jpg_path = "./data_science_bowl_2017/000JPG/000JPGALL/%s.jpg" % filename
            convert_from_dicom_to_jpg(img_array, low, high, output_jpg_path)

    else:
        # 用python批量把DICOM(dcm)转换成jpg图像  https://blog.csdn.net/you2336/article/details/107615846
        for j in range(len(file)):
            # if j >= 16:
            #     continue
            # file.sort()  # 文件列表按名称排序
            threeD_patient = load_scan(threeD_file_path)
            threeD_patient_pixels = get_pixels_hu(threeD_patient)
            print("other")
            # print("file", file[j])
            filename = file[j][25:][:-4]  # 文件名
            # print("filename", filename)
            # python 截取数据到 斜杆/      https://blog.csdn.net/m0_46400195/article/details/125599965
            list2 = filename.split('\\')  # 采用"."进行分割       \\
            # print("list2[0]", list2[0])
            if not os.path.exists('./data_science_bowl_2017/000JPG/005JPG/' + list2[0]):
                os.mkdir('./data_science_bowl_2017/000JPG/005JPG/' + list2[0])

            filename = list2[0] + '/' + str(j)            # 这样省去重新命名     只取16个，1-16的命名方式
            # # save("./data_science_bowl_2017/000JPG/005JPG/%s.jpg" % filename)
            # if j == 0:
            #     save("./data_science_bowl_2017/000JPG/005JPG/%s.jpg" % (list2[0] + '/' + str(0)))

            # ds = pydicom.read_file(file[j])          #读取文件
            # img = ds.pixel_array
            # by YXY
            img = threeD_patient_pixels[j]

            imageio.imwrite("./data_science_bowl_2017/000JPG/005JPG/%s.jpg" % filename, img)

            # 修改过后的批量转换且无纯黑图像问题     https://blog.csdn.net/ambrace/article/details/124662370
            ds_array = sitk.ReadImage("./data_science_bowl_2017/000JPG/005JPG/%s.jpg" % filename)
            img_array = sitk.GetArrayFromImage(ds_array)
            shape = img_array.shape
            img_array = np.reshape(img_array, (shape[0], shape[1]))  # 获取array中的height和width
            high = np.max(img_array)
            low = np.min(img_array)
            output_jpg_path = "./data_science_bowl_2017/000JPG/005JPG/%s.jpg" % filename
            convert_from_dicom_to_jpg(img_array, low, high, output_jpg_path)



