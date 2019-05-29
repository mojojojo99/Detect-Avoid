import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_in = pd.read_csv('/Users/jolim/Desktop/filter/model/data_BW1_withoutlabel')
a = np.array(data_in)
rows = (a.size[0])
data = np.zeros(rows, 15)
labels = np.zeros(rows, 1)
numFrames = 5



for i in range (numFrames - 1, rows):
	data[i, 0:3] = a[i, 0:3]
	data[i, 3:6] = a[i-1, 0:3]
	data[i, 6:9] = a[i-2, 0:3]
	data[i, 9:12] = a[i-3, 0:3]
	data[i, 12:15] = a[i-4, 0:3]


print (a)
plt.plot(a)
plt.show()
