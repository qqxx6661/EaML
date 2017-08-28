import numpy as np
import matplotlib.pyplot as plt

size = 4
x = np.arange(size)

# video_file_cloud = (56780, 67513, 95846, 124214)  # real
video_file_cloud = (4.754, 4.829, 4.982, 5.094)
# video_file_edge = (54125, 66182, 66182, 66128)  # f, 各自处理1,2,2,2个摄像头平均值
video_file_edge = (4.733, 4.821, 4.821, 4.821)
# prediction_cloud = (49, 90, 143, 186)  # f
prediction_cloud = (1.69, 1.957, 2.155, 2.267)
# prediction_EaOP = (49, 90, 143, 186)  # f
prediction_EaOP = (1.69, 1.957, 2.155, 2.267)

total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2

plt.xlabel('Total Camera Numbers', fontsize=18)
plt.ylabel('Computing Cost (lg(ms))', fontsize=18)


plt.bar(x-0.45*width, video_file_cloud, fc='#036564', width=0.75*width, label='Video Analysis (Cloud)')
plt.bar(x-0.45*width, prediction_cloud, fc='#033649', width=0.75*width, bottom=video_file_cloud, label='Prediction (Cloud)')
plt.bar(x+0.45*width, video_file_edge, fc='#764D39', width=0.75*width, label='Video Analysis (EaOP)')
plt.bar(x+0.45*width, prediction_EaOP, fc='#250807', width=0.75*width, bottom=video_file_edge, label='Prediction (EaOP)')

plt.xticks(x, (2, 4, 6, 8), fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='center', bbox_to_anchor=(0.77, 0.13), fontsize=11)
plt.show()
