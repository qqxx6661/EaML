#!/usr/bin/env python3
# coding=utf-8
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2

# 从6个csv中循环读取数据，组成当前帧数据
for i in range(1800):
    for j in range(6):

# 在每一帧的循环中，将该帧出现的人与画廊进行匹配，若与某人相似度大于某值，则认为是那个人,
# 写入那个人的person_id.csv,[#frame, #camid, #[xywh]]
# 若没有匹配或者画廊里还没有，则加入画廊，给一个新的person_ID

# 预测最后方向，循环打开person_id.csv,最后N帧，预测