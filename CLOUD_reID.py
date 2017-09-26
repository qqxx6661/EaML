#!/usr/bin/env python3
# coding=utf-8
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2

def reID_result(image_HSV):


    return person_ID

results = {}
# target
target_image_path = '/reID_image_test/image5.jpg'
target_image = cv2.imread(target_image_path)
target_image_hist = cv2.calcHist([target_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
target_image_hist = cv2.normalize(target_image_hist, target_image_hist).flatten()

# gallery
for i in range(1, 161):
    gallery_image_path = '/reID_image_test/image' + str(i) + '.jpg'
    gallery_image = cv2.imread(gallery_image_path)
    gallery_image_hist = cv2.calcHist([gallery_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    gallery_image_hist = cv2.normalize(gallery_image_hist, gallery_image_hist).flatten()
    d = dist.euclidean(target_image_hist, gallery_image_hist)
    results[i] = d

print(results)
