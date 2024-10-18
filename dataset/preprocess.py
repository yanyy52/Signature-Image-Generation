import itertools

import cv2
import numpy as np
import torch
from PIL import Image
import os
import pandas as pd
import random

from matplotlib import pyplot as plt
from scipy import ndimage

def remove_thresholding(dir):
    image = cv2.imread(dir)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用最佳阈值进行二值化
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV)

    # plt.figure()
    # plt.imshow(binary_image)
    # plt.show()

    # 查找图像的轮廓
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 计算每个轮廓的边界框，并将它们合并为一个大的边界框
    if contours:
        merged_x = min([cv2.boundingRect(contour)[0] for contour in contours])
        merged_y = min([cv2.boundingRect(contour)[1] for contour in contours])
        merged_w = max(
            [cv2.boundingRect(contour)[0] + cv2.boundingRect(contour)[2] for contour in contours]) - merged_x
        merged_h = max(
            [cv2.boundingRect(contour)[1] + cv2.boundingRect(contour)[3] for contour in contours]) - merged_y

        # 裁剪图像以去掉四周的多余白色部分
        cropped_image = image[int(merged_y):int(merged_y + merged_h),
                        int(merged_x):int(merged_x + merged_w)]

    # plt.figure()
    # plt.imshow(cropped_image)
    # plt.show()

    return cropped_image

def otsu_erzhi(dir):
    # # MSDS专用
    # # Step 1: Load the four-channel image
    # four_channel_img = cv2.imread(dir, -1)
    # # plt.figure()
    # # plt.imshow(four_channel_img)
    # # plt.show()
    #
    # # Step 2: Convert the four-channel image to grayscale
    # gray_img = cv2.cvtColor(four_channel_img[:, :, 1:], cv2.COLOR_BGR2GRAY)
    # # gray_img[gray_img == 0] = 255
  
    gray_image = cv2.imread(dir, 0)
    # plt.figure()
    # plt.imshow(gray_image)
    # plt.show()

    # 计算每个灰度级别的像素数量
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    # 归一化直方图
    hist_norm = hist.ravel() / hist.max()
    # 初始化类内方差之和和最佳阈值
    var_within_class = np.zeros((256,))
    # 计算类内方差之和
    for threshold in range(256):
        # 类别1和类别2的像素数量
        w1 = np.sum(hist_norm[:threshold])
        w2 = np.sum(hist_norm[threshold:])
        # 类别1和类别2的像素均值
        mu1 = np.sum(np.arange(0, threshold) * hist_norm[:threshold]) / (w1 + 1e-10)
        mu2 = np.sum(np.arange(threshold, 256) * hist_norm[threshold:]) / (w2 + 1e-10)
        # 类内方差之和
        var_within_class[threshold] = w1 * w2 * ((mu1 - mu2) ** 2)
    # 选择使得类内方差之和最小的阈值作为最佳阈值
    optimal_threshold = np.argmax(var_within_class)
    # 使用最佳阈值进行二值化
    _, binary_image = cv2.threshold(gray_image, optimal_threshold, 255, cv2.THRESH_BINARY)
    # binary_image = 255 - binary_image
    # binary_image = np.where(gray_image > optimal_threshold, 255, gray_image)

    image_three_channels = cv2.merge([binary_image] * 3)


    # plt.figure()
    # plt.imshow(image_three_channels)
    # plt.show()


    return image_three_channels
  
def pro_CEDAR(root, w=220, h=155):
    # if not os.path.exists(f'{root}_pro'):
    #     os.mkdir(f'{root}_pro')
    # else:
    #     print(f"目录 '{f'{root}_pro'}' 已经存在。")

    for filename in os.listdir(root):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            dir =f'{root}/{filename}'  
            # image = otsu_erzhi(dir)
            cv2.imwrite(f'{root}_pro/{filename}', image)
            # with Image.open(f'{root}/{filename}') as img:
            #     img = img.resize((w, h))
            #     img.save(
            #         f'{root}_resize/{filename}')


def generate_pairs_CEDAR(root: str, cutting_point: int):
    size = 55
    # num_genuine = 24
    num_genuine = 48
    num_forged = 24

    def pair_string_genuine(i, j, k):
        # return f'full_org_resize/original_{i}_{j}.png full_org_resize/original_{i}_{k}.png 1\n'

    def pair_string_forged(i, j, k):
        # return f'full_org_resize/original_{i}_{j}.png full_forg_resize/forgeries_{i}_{k}.png 0\n'

    def generate(file, i):
        # reference-genuine pairs
        for j in range(0, 24):
             for k in range(j + 1, num_genuine + 1):
                file.write(pair_string_genuine(i, j, k))
        # reference-forgered pairs
        org_forg = [(j, k) for j in range(0, num_genuine + 1)
                    for k in range(1, num_forged + 1)]
        for (j, k) in random.choices(org_forg, k=276):
        # for (j, k) in random.choices(org_forg, k=1128):
            file.write(pair_string_forged(i, j, k))

    # with open(f'{root}/train_pairs.txt', 'w') as f:
    #     for i in range(1, cutting_point):
    #         generate(f, i)

    with open(f'{root}/test_pairs.txt', 'w') as f:
        for i in range(cutting_point, size + 1):
            generate(f, i)

def generate_pairs_BHSig(root: str, cutting_point: int):
    size = 100  #160 100
    num_genuine = 48 #24 48
    num_forged = 30

    def pair_string_genuine(i, j, k):
        return f'{i:03}/B-S-{i}-G-{j:02}.png {i:03}/B-S-{i}-G-{k:02}.png 1\n'


    def pair_string_forged(i, j, k):
        return f'{i:03}/B-S-{i}-G-{j:02}.png {i:03}/B-S-{i}-F-{k:02}.png 0\n'

    def generate(file, i):
        # reference-genuine pairs
        for j in range(0, num_genuine):
            for k in range(j + 1, num_genuine):
                file.write(pair_string_genuine(i, j, k))
        # reference-forgered pairs
        org_forg = [(j, k) for j in range(0, num_genuine)
                    for k in range(1, num_forged + 1)]
        for (j, k) in random.choices(org_forg, k=276):
        # for (j, k) in random.choices(org_forg, k=1128):
            file.write(pair_string_forged(i, j, k))

    with open(f'{root}/train_pairs.txt', 'w') as f:
        for i in range(1, cutting_point):
            generate(f, i)

    # with open(f'{root}/test_pairs.txt', 'w') as f:
    #     for i in range(cutting_point, size + 1):
    #         generate(f, i)



def generate_MSDS_pairs(root: str, cutting_point: int):
    size = 402

    num_genuine = 20
    num_forged=10

    def pair_string_genuine(i, j, k):
        return f'{i}/g_{i}_{j}.png {i}/g_{i}_{k}.png 1\n'

    def pair_string_forged(i, j, k):
        return f'{i}/g_{i}_{j}.png {i}/f_{i}_{k}.png 0\n'

    def generate(file, i):
        for j in range(0, 10):
            for k in range(j + 1, 10):
                file.write(pair_string_genuine(i, j, k))
        # reference-forgered pairs
        org_forg = [(j, k) for j in range(0, 10)
                    for k in range(0, 10)]
        for (j, k) in org_forg:
        # for (j, k) in random.choices(org_forg, k=45):
            file.write(pair_string_forged(i, j, k))


    # with open(f'{root}/pairs.txt', 'w') as f:
    #     for i in range(0, cutting_point):
    #         generate(f, i)

    # with open(f'{root}/test_pairs.txt', 'w') as f:
    #     for i in range(cutting_point, size):
    #         generate(f, i)

if __name__ == '__main__':
    # pro_CEDAR('CEDAR/full_org')
    # pro_CEDAR('CEDAR/full_forg')
    # generate_pairs_CEDAR('CEDAR/', 51)

    # pro_BHSig('BHSig260/Bengali', 100)
    # generate_pairs_BHSig('BHSig260/Bengali', 81)
    # resize_BHSig_img('dataset/BHSig260/Hindi', 160)
    # pro_BHSig('BHSig260/Hindi_pro', 160)
    # generate_pairs_Hindi('BHSig260/Hindi', 129)
  
    # generate_SigComp_pairs('SigComp2011')
    # pro_CEDAR('SigComp2011/train/Offline_Genuine')

    # pro_MSDS('MSDS/session1')
    # generate_MSDS_pairs('MSDS/session1', 321)
