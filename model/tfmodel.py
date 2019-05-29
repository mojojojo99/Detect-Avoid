import tensorflow as tf
from tensorflow.keras import layers
import numpy as np 
import pandas as pd

print(tf.VERSION)
print(tf.keras.__version__)

data = pd.read_csv('/Users/jolim/Desktop/filter/model/data')




model = tf.keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(layers.Dense(64, activation='relu'))
# Add another:
model.add(layers.Dense(64, activation='relu'))
# Add a softmax layer with 1 output units:
model.add(layers.Dense(1, activation='softmax'))



#################################################################
model = tf.keras.Sequential([
# Adds a densely-connected layer with 64 units to the model:
layers.Dense(64, activation='relu', input_shape=(3,)),
# Add another:
layers.Dense(64, activation='relu'),
# Add a softmax layer with 10 output units:
layers.Dense(1, activation='softmax')])

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])


data = np.random.random((1000, 3))
labels = np.random.random((1000, 1))

model.fit(data, labels, epochs=10, batch_size=32)


