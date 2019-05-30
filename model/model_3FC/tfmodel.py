import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import matplotlib.pyplot as plt
import pylab

print(tf.VERSION)
print(tf.keras.__version__)

data = np.load('/Users/jolim/Desktop/filter/model/data.npy')
labels = np.load('/Users/jolim/Desktop/filter/model/labels.npy')

restart = True



model = tf.keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(layers.Dense(64, activation='relu'))
# Add another:
model.add(layers.Dense(64, activation='relu'))
# Add another:
model.add(layers.Dense(64, activation='relu'))
# Add a sigmoid layer with 1 output unit:
model.add(layers.Dense(1, activation='sigmoid'))



#################################################################
model = tf.keras.Sequential([
# Adds a densely-connected layer with 64 units to the model:
layers.Dense(64, activation='relu', input_shape=(15,)),
# Add another:
layers.Dense(64, activation='relu'),
# Add another:
layers.Dense(64, activation='relu'),
# Add a sigmoid layer with 1 output unit:
layers.Dense(1, activation='sigmoid')])

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])


history = model.fit(data, labels, epochs=1000, batch_size=32, shuffle=True)
model.save_weights('./checkpoints/my_checkpoint')

exists = os.path.isfile('./checkpoints/losses.npy')
if (not  restart and exists):
    prev_loss = np.load('./checkpoints/losses.npy')
    prev_acc = np.load('./checkpoints/acc.npy')

    all_loss = np.append(prev_loss, np.array(history.history ["loss"]))
    all_acc = np.append(prev_loss, np.array(history.history ["acc"]))
else:
    all_loss = np.array(history.history ["loss"])
    all_acc =  np.array(history.history ["acc"])


np.save ('./checkpoints/losses', all_loss)
np.save ('./checkpoints/acc', all_acc)

t = np.arange(0,all_loss.shape[0])
fig = plt.figure()
plt.plot(t, all_loss, 'r', label = 'loss')
#plt.plot(t, all_acc, 'b', label = 'accuracy')
plt.show()
