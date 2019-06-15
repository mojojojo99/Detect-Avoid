import numpy as np
import keras
from PIL import Image

from model import SegNet

import dataset

height = 360
width = 480
classes = 12
epochs = 100
batch_size = 1
log_filepath='./logs_100/'

data_shape = 360*480

def writeImage(image, filename):
    """ label data to colored image """
    Sky = [128,128,128]
    Building = [128,0,0]
    Pole = [192,192,128]
    Road_marking = [255,69,0]
    Road = [128,64,128]
    Pavement = [60,40,222]
    Tree = [128,128,0]
    SignSymbol = [192,128,128]
    Fence = [64,64,128]
    Car = [64,0,128]
    Pedestrian = [64,64,0]
    Bicyclist = [0,128,192]
    Unlabelled = [0,0,0]
    r = image.copy()
    g = image.copy()
    b = image.copy()
    label_colours = np.array([Sky, Building, Pole, Road_marking, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])
    for l in range(0,12):
        r[image==l] = label_colours[l,0]
        g[image==l] = label_colours[l,1]
        b[image==l] = label_colours[l,2]
    rgb = np.zeros((image.shape[0], image.shape[1], 3))
    rgb[:,:,0] = r/1.0
    rgb[:,:,1] = g/1.0
    rgb[:,:,2] = b/1.0
    im = Image.fromarray(np.uint8(rgb))
    im.save(filename)

def predict(test):
    model = keras.models.load_model('seg_100.h5')
    probs = model.predict(test, batch_size=1)

    prob = probs[0].reshape((height, width, classes)).argmax(axis=2)
    return prob

def main():
    print("loading data...")
    ds = dataset.Dataset(test_file='val.txt', classes=classes)
    test_X, test_y = ds.load_data('test') # need to implement, y shape is (None, 360, 480, classes)
    # test_X = ds.load_data2('test')

    test_X = ds.preprocess_inputs(test_X)
    test_Y = ds.reshape_labels(test_y)

    prob = predict(test_X)
    writeImage(prob, 'val.png')

if __name__ == '__main__':
    main()
