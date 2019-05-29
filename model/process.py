import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
numFrames = 5
data_in = pd.read_csv('/Users/jolim/Desktop/filter/model/data_BW1_withoutlabel')
a = np.array(data_in)
rows = (a.shape[0])
data = np.zeros([rows-numFrames + 1, numFrames*3])
labels = np.zeros([rows- numFrames + 1, 1])




for i in range (numFrames - 1, rows):
	for j in range(numFrames):
		print (data[i-numFrames, j:(j+1)*3])
		print (a[(i-j), 0:3])
		data[i-numFrames, j:(j+1)*3] = a[i-j, 0:3]
	labels[i-numFrames] = a[i, 3]

print (data)
np.save('data', data)
np.save('labels', labels)
