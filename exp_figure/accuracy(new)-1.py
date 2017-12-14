import numpy as np
import matplotlib.pyplot as plt
import math

size = 6
x = np.arange(size)

Learn_180 = [75.19, 69.92, 69.92, 58.65, 58.65, 42.11]

Learn_360 = [81.95, 78.20, 69.92, 66.92, 65.41, 40.6]

Person_720 = [82.71, 78.95, 74.44, 51.88, 48.87, 38.35]

total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2

plt.xlabel('Prediction delay (s)', fontsize=20)
plt.ylabel('Average accuracy (%)', fontsize=20)
plt.bar(x-0.9*width, Learn_180, fc='#faa755', width=0.75*width, label='180s training model')
plt.bar(x, Learn_360, fc='#6b473c', width=0.75*width, label='360s training model')
plt.bar(x+0.9*width, Person_720, fc='#8a5d19', width=0.75*width, label='720s training model')
plt.xticks(x, ('1/3', 1, 3, 5, 10, 15), fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='lower right', fontsize=15)
plt.show()
