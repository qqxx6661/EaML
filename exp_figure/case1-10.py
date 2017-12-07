import numpy as np
import matplotlib.pyplot as plt
import math

size = 3
x = np.arange(size)

Person_0 = [35.59, 29.26, 25.668]

Person_1 = [96.949, 94.18, 90.731]

Person_2 = [95.254, 85.542, 85.383]

total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2

plt.xlabel('Prediction delay (frame)', fontsize=20)
plt.ylabel('Accuracy of Prediction (%)', fontsize=20)
plt.bar(x-0.9*width, Person_0, fc='#faa755', width=0.75*width, label='Person 0')
plt.bar(x, Person_1, fc='#6b473c', width=0.75*width, label='Person 1')
plt.bar(x+0.9*width, Person_2, fc='#8a5d19', width=0.75*width, label='Person 2')
plt.xticks(x, (1, 10, 30), fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='upper right', fontsize=15)
plt.show()
