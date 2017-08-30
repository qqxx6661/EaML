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

transmission_cloud = [1920, 4041, 59452, 79562]  # μs per frame 2,4,6,8个摄像头每帧
log(transmission_cloud)

transmission_EaOP = [8.768, 17.536, 35.072, 70.144]
log(transmission_EaOP)

video_file_cloud = [56510000, 67513000, 95846000, 124214000]  # cloud处理2,4,6,8个摄像头（30s）
prediction_cloud = [56000, 108000, 175000, 257000]
computation_cloud = [56566000, 67621000, 96021000, 124471000]  # 上面两数组相加
ave(computation_cloud)
log(computation_cloud)

video_file_EaOP = [54125000, 65182000, 86251000, 101381000]  # edge各自处理1,2,3,4个摄像头（30s）
prediction_EaOP = [49000, 90000, 143000, 186000]
computation_EaOP = [54174000, 65272000, 86394000, 101567000]  # 上面两数组相加
ave(computation_EaOP)
log(computation_EaOP)

total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2

plt.xlabel('Total Camera Numbers', fontsize=20)
plt.ylabel('Overall Cost (lg(μs))', fontsize=20)


plt.bar(x-0.45*width, transmission_cloud, fc='#036564', width=0.75*width, label='Transmission (Cloud)')
plt.bar(x-0.45*width, computation_cloud, fc='#033649', width=0.75*width, bottom=transmission_cloud, label='Computation (Cloud)')
plt.bar(x+0.45*width, transmission_EaOP, fc='#764D39', width=0.75*width, label='Transmission (EaOP)')
plt.bar(x+0.45*width, computation_EaOP, fc='#250807', width=0.75*width, bottom=transmission_EaOP, label='Computation (EaOP)')

plt.xticks(x, (2, 4, 6, 8), fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='center', bbox_to_anchor=(0.75, 0.15), fontsize=17)
plt.show()
