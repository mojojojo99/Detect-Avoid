import os
import glob
import numpy as np
import keras

from model import SegNet

import dataset

input_shape = (360, 480, 3)
classes = 12
epochs = 100
batch_size = 1
log_filepath='./logs/'

data_shape = 360*480

class_weighting = [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]

## set gpu usage
import tensorflow as tf
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction = 0.8))
session = tf.Session(config=config)
keras.backend.tensorflow_backend.set_session(session)

def main():
    print("loading data...")
    ds = dataset.Dataset(classes=classes)
    train_X, train_y = ds.load_data('train') # need to implement, y shape is (None, 360, 480, classes)

    train_X = ds.preprocess_inputs(train_X)
    train_Y = ds.reshape_labels(train_y)
    print("input data shape...", train_X.shape)
    print("input label shape...", train_Y.shape)

    test_X, test_y = ds.load_data('test') # need to implement, y shape is (None, 360, 480, classes)
    test_X = ds.preprocess_inputs(test_X)
    test_Y = ds.reshape_labels(test_y)

    tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, histogram_freq=1, write_graph=True, write_images=True)
    print("creating model...")
    model = SegNet(input_shape=input_shape, classes=classes)
    model.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])

    model.fit(train_X, train_Y, batch_size=batch_size, epochs=epochs,
              verbose=1, class_weight=class_weighting , validation_data=(test_X, test_Y), shuffle=True
              , callbacks=[tb_cb])

    model.save('seg.h5')

if __name__ == '__main__':
    main()
