import tensorflow as tf
from tensorflow.keras import layers
import numpy as np 

print(tf.VERSION)
print(tf.keras.__version__)

data = np.load('/Users/jolim/Desktop/filter/model/data.npy')
labels = np.load('/Users/jolim/Desktop/filter/model/labels.npy')




model = tf.keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(layers.Dense(64, activation='relu'))
# Add another:
model.add(layers.Dense(64, activation='relu'))
# Add another:
model.add(layers.Dense(64, activation='relu'))
# Add another:
model.add(layers.Dense(64, activation='relu'))
# Add another:
model.add(layers.Dense(64, activation='relu'))
# Add a softmax layer with 1 output unit:
model.add(layers.Dense(1, activation='sigmoid'))



#################################################################
model = tf.keras.Sequential([
# Adds a densely-connected layer with 64 units to the model:
layers.Dense(64, activation='relu', input_shape=(15,)),
# Add another:
layers.Dense(64, activation='relu'),
# Add another:
layers.Dense(64, activation='relu'),
# Add another:
layers.Dense(64, activation='relu'),
# Add another:
layers.Dense(64, activation='relu'),
# Add a softmax layer with 1 output unit:
layers.Dense(1, activation='sigmoid')])

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])


model.fit(data, labels, epochs=10, batch_size=32, validation_data=(data, labels))
model.save_weights('./checkpoints/my_checkpoint2')



