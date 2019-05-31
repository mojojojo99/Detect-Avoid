import matplotlib.pyplot as plt
import numpy as np


label = np.load('./label_sigmoid.npy')
predict = np.load('./predict_sigmoid.npy')

y = label[800:1000]
p = predict[800:1000]
plt.plot(p)
plt.plot(y)
plt.title('Prediction')
plt.legend(['predicted', 'actual'])
plt.ylabel('Predicted value')
plt.xlabel('time')
plt.show()
