import numpy as np
import matplotlib.pyplot as plt

size = 4
x = np.arange(size)

# video_file = (11132, 21164, 34452, 45208)  # all real
video_file = (4.047, 4.326, 4.537, 4.655)

# data_to_cam = (1.602, 1.908, 2.1, 2.217)
# data_to_cam = (40, 80, 120, 160)

# data_to_cloud = (127, 248, 365, 488)
data_to_cloud = (2.103, 2.394, 2.562, 2.688)

total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2

plt.xlabel('Total Camera Numbers', fontsize=18)
plt.ylabel('Communication Cost (lg(Byte))', fontsize=18)
plt.bar(x-0.45*width, video_file, fc='#036564', width=0.75*width, label='Input Data to Cloud (Cloud)')
# plt.bar(x-0.45*width, data_to_cam, fc='#033649', width=0.75*width, bottom=video_file, label='Feedback (Cloud)')
plt.bar(x+0.45*width, data_to_cloud, fc='#764D39', width=0.75*width, label='Input Data to Cloud (EaOP)')
# plt.bar(x+0.45*width, data_to_cam, fc='#250807', width=0.75*width, bottom=data_to_cloud, label='Feedback (EaOT)')
plt.xticks(x, (2, 4, 6, 8), fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='center', bbox_to_anchor=(0.77, 0.08), fontsize=11)
plt.show()
