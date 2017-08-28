import numpy as np
import matplotlib.pyplot as plt

size = 4
x = np.arange(size)
# transmission_cloud = (18.08, 33.86, 55.12, 72.33)  # real, ms per frame, but ms not enough
transmission_cloud = (3.312, 3.613, 3.789, 3.914)

# transmission_EaOP = (0.2, 1.663, 1.833, 1.959)
transmission_EaOP = (1.301, 1.663, 1.833, 1.959)

# prediction_cloud = (49, 90, 143, 186)  # f
prediction_cloud = (1.69, 1.957, 2.155, 2.267)
# prediction_EaOP = (49, 90, 143, 186)  # f
prediction_EaOP = (1.69, 1.957, 2.155, 2.267)



total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2

plt.xlabel('Total Camera Numbers', fontsize=18)
plt.ylabel('Overall Cost (lg(ms))', fontsize=18)


plt.bar(x-0.45*width, transmission_cloud, fc='#036564', width=0.75*width, label='Transmission (Cloud)')
plt.bar(x-0.45*width, prediction_cloud, fc='#033649', width=0.75*width, bottom=transmission_cloud, label='Prediction (Cloud)')
plt.bar(x+0.45*width, transmission_EaOP, fc='#764D39', width=0.75*width, label='Transmission (EaOP)')
plt.bar(x+0.45*width, prediction_EaOP, fc='#250807', width=0.75*width, bottom=transmission_EaOP, label='Prediction (EaOP)')

plt.xticks(x, (2, 4, 6, 8), fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='center', bbox_to_anchor=(0.79, 0.13), fontsize=11)
plt.show()
