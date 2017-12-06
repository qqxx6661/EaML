import numpy as np
import matplotlib.pyplot as plt
import math

size = 3
x = np.arange(size)

Person_1 = [94.237, 93.46, 86.988]

Person_2 = [77.288, 74.355, 73.322]

total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2

plt.xlabel('Prediction delay (frame)', fontsize=20)
plt.ylabel('Accuracy of Prediction (%)', fontsize=20)
plt.bar(x-0.45*width, Person_1, fc='#036564', width=0.75*width, label='Person 1')
plt.bar(x+0.45*width, Person_2, fc='#764D39', width=0.75*width, label='Person 2')
plt.xticks(x, (1, 10, 30), fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='lower right', fontsize=15)
plt.show()
