import matplotlib.pyplot as plt
import numpy as np
from keras import Input, Model
from keras.layers import Dense
from keras import initializers
import pandas as  pd
from tcn import TCN
import os
restart  = True


data_in = np.array(pd.read_csv('/Users/jolim/Desktop/filter/model/30May/data.csv'))
data = data_in[:,  0:3]
labels = data_in[:, 3]

lookback_window = 7


x, y = [], []
for i in range(lookback_window - 1, len(data)):
      x.append(data[i - lookback_window + 1:i+1])
      y.append(labels[i])
x = np.array(x)
y = np.array(y)

print(x.shape)
print(y.shape)

i = Input(shape=(lookback_window, 3))
m = TCN(nb_filters=3, kernel_size=7,  nb_stacks=1, padding='causal', use_skip_connections=True, dropout_rate=0.0, return_sequences=False, name='tcn')(i)
m = Dense(1, activation='sigmoid')(m)

model = Model(inputs=[i], outputs=[m])

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print('Train...')
history = model.fit(x, y, validation_split= 0.2,  epochs=300, verbose=2)
#model.save_weights('./checkpoints/my_checkpoint')

all_loss = np.array(history.history ["loss"])
all_acc =  np.array(history.history ["acc"])


#np.save ('./checkpoints/losses', all_loss)
#np.save ('./checkpoints/acc', all_acc)

t = np.arange(0,all_loss.shape[0])
p = model.predict(x)
np.save('predict_sigmoid', p)
np.save('label_sigmoid', y)
plt.plot(p)
plt.plot(y)
plt.title('Prediction')
plt.legend(['predicted', 'actual'])
plt.show()
