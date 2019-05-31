import matplotlib.pyplot as plt
import numpy as np
from keras import Input, Model
from keras.layers import Dense
import pandas as  pd
from tcn import TCN


data_in = np.array(pd.read_csv('/Users/jolim/Desktop/filter/model/30May/data.csv'))
data = data_in[:,  0:3]
labels = data_in[:, 3]

lookback_window = 12  # months.


x, y = [], []
for i in range(lookback_window - 1, len(data)):
      x.append(data[i - lookback_window + 1:i+1])
      y.append(labels[i])
x = np.array(x)
y = np.array(y)

print(x.shape)
print(y.shape)

i = Input(shape=(lookback_window, 3))
m = TCN(nb_filters=64, kernel_size=2, nb_stacks=1, padding='causal', use_skip_connections=True, dropout_rate=0.0, return_sequences=False, name='tcn')(i)
m = Dense(1, activation='linear')(m)

model = Model(inputs=[i], outputs=[m])

model.summary()

model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])

print('Train...')
model.fit(x, y, epochs=100, verbose=2)

p = model.predict(x)
np.save('predict_sigmoid', p)
np.save('label_sigmoid', y)
plt.plot(p)
plt.plot(y)
plt.title('Prediction')
plt.legend(['predicted', 'actual'])
plt.show()
