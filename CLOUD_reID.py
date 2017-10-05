#!/usr/bin/env python3
# coding=utf-8
from scipy.spatial import distance as dist
import numpy as np
import cv2

def reID(filename, gallery_person_list):
    person = 0
    target_hist = np.load(filename)

    # max_similiar = 0
    # for i in range(len(gallery_person_list)):
    #     gallery_hist = np.load(gallery_person_list[i])
    #     # print(target_hist)
    #     similiar = cv2.compareHist(target_hist, gallery_hist, cv2.HISTCMP_CORREL)
    #     print(filename, gallery_person_list[i], similiar)
    #     if similiar == 0.0:  # 写入失败文件忽略
    #         print('图像数据损坏，不予处理')
    #         return -1
    #     if max_similiar < similiar:
    #         max_similiar = similiar
    #         person = i

    # min_similiar = 99999
    # for i in range(len(gallery_person_list)):
    #     gallery_hist = np.load(gallery_person_list[i])
    #     # print(target_hist)
    #     similiar = cv2.compareHist(target_hist, gallery_hist, cv2.HISTCMP_CHISQR)
    #     print(filename, gallery_person_list[i], similiar)
    #     if similiar == 0.0:  # 写入失败文件忽略
    #         print('图像数据损坏，不予处理')
    #         return -1
    #     if min_similiar > similiar:
    #         min_similiar = similiar
    #         person = i

    min_similiar = 99999
    for i in range(len(gallery_person_list)):
        gallery_hist = np.load(gallery_person_list[i])
        # print(target_hist)
        similiar = dist.cityblock(gallery_hist, target_hist)
        print(filename, gallery_person_list[i], similiar)
        if similiar == 0.0:  # 写入失败文件忽略
            print('图像数据损坏，不予处理')
            return -1
        if min_similiar > similiar:
            min_similiar = similiar
            person = i

    return person


# reID('BSU_data/0/61_0.npy', ['gallery/133_0.npy', 'gallery/60_0.npy'])

# print(np.load('BSU_data/0/95_0.npy'))