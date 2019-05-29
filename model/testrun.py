import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/jolim/Desktop/filter/model/data')
a = np.array(data)
a[:, 2] = a[:, 2]/100
print (a)
plt.plot(a)
plt.show()
