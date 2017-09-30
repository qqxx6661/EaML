#!/usr/bin/env python3
# coding=utf-8
from scipy.spatial import distance as dist
import numpy as np
import cv2

def reID(filename, gallery_person_list):

    max_similiar = 0
    person = 0
    target_hist = np.load(filename)
    for i in range(len(gallery_person_list)):
        gallery_hist = np.load(gallery_person_list[i])
        # print(target_hist)
        similiar = cv2.compareHist(target_hist, gallery_hist, cv2.HISTCMP_INTERSECT)
        print(i, similiar)
        if max_similiar < similiar:
            max_similiar = similiar
            person = i
    # print(person)
    return person


# reID('BSU_data/0/61_0.npy', ['gallery/133_0.npy', 'gallery/60_0.npy'])

# print(np.load('BSU_data/0/95_0.npy'))