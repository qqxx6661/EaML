import numpy as np
import matplotlib.pyplot as plt
import math


def log(list_name):
    for i in range(len(list_name)):
        list_name[i] = math.log10(list_name[i])
        print(list_name[i])
    return list_name


def ave(list_name):
    for i in range(len(list_name)):
        list_name[i] = list_name[i] / 900
        print(list_name[i])
    return list_name

size = 4
x = np.arange(size)

video_file_cloud = [56510000, 67513000, 95846000, 124214000]  # cloud处理2,4,6,8个摄像头（30s）
video_file_cloud = ave(video_file_cloud)
log(video_file_cloud)

# video_file_edge = [54125000, 65182000, 86251000, 101381000]  # edge各自处理1,2,3,4个摄像头（30s）
# ave(video_file_edge)
# log(video_file_edge)

prediction_cloud = [56000, 108000, 175000, 257000]  # cloud预测30s视频所用时间，分析视频后预测
prediction_cloud = ave(prediction_cloud)
prediction_cloud = log(prediction_cloud)

prediction_EaOP = [49000, 90000, 143000, 186000]  # cloud预测30s视频所用时间，拿到数据后直接预测
prediction_EaOP = ave(prediction_EaOP)
prediction_EaOP = log(prediction_EaOP)

total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2

plt.xlabel('Total Camera Numbers', fontsize=19)
plt.ylabel('Computational Cost at Cloud (lg(μs))', fontsize=19)


plt.bar(x-0.45*width, video_file_cloud, fc='#036564', width=0.75*width, label='Object Detection (Cloud)')
plt.bar(x-0.45*width, prediction_cloud, fc='#033649', width=0.75*width, bottom=video_file_cloud, label='Prediction (Cloud)')
# plt.bar(x+0.45*width, video_file_edge, fc='#764D39', width=0.75*width, label='Object Detection (EaOP)')
plt.bar(x+0.45*width, prediction_EaOP, fc='#250807', width=0.75*width, label='Prediction (EaOP)')

plt.xticks(x, (2, 4, 6, 8), fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='center', bbox_to_anchor=(0.7, 0.11), fontsize=16)
plt.show()
