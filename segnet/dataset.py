import cv2
import numpy as np

from keras.applications import imagenet_utils

import os

DataPath = './CamVid/'
data_shape = 360*480

class Dataset:
    def __init__(self, classes=12, train_file='train.txt', test_file='test.txt'):
        self.train_file = train_file
        self.test_file = test_file
        self.data_shape = 360*480
        self.classes = classes

    def normalized(self, rgb):
        #return rgb/255.0
        norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)

        b=rgb[:,:,0]
        g=rgb[:,:,1]
        r=rgb[:,:,2]

        norm[:,:,0]=cv2.equalizeHist(b)
        norm[:,:,1]=cv2.equalizeHist(g)
        norm[:,:,2]=cv2.equalizeHist(r)

        return norm

    def one_hot_it(self, labels):
        x = np.zeros([360,480,12])
        for i in range(360):
            for j in range(480):
                x[i,j,labels[i][j]] = 1
        return x

    def load_data(self, mode='train'):
        data = []
        label = []
        if (mode == 'train'):
            filename = self.train_file
        else:
            filename = self.test_file

        with open(DataPath + filename) as f:
            txt = f.readlines()
            txt = [line.split(' ') for line in txt]

        for i in range(len(txt)):
            data.append(self.normalized(cv2.imread(os.getcwd() + txt[i][0][7:])))
            label.append(self.one_hot_it(cv2.imread(os.getcwd() + txt[i][1][7:][:-1])[:,:,0]))
            print('.',end='')
        #print("train data file", os.getcwd() + txt[i][0][7:])
        #print("label data raw", cv2.imread(os.getcwd() + '/CamVid/trainannot/0001TP_006690.png'))
        return np.array(data), np.array(label)

    def load_data2(self, mode='test'):
        data = []
        if (mode == 'train'):
            filename = self.train_file
        else:
            filename = self.test_file

        with open(DataPath + filename) as f:
            txt = f.readlines()
            txt = [line.split(' ') for line in txt]

        for i in range(1):
            data.append(self.normalized(cv2.imread(os.getcwd() + txt[i][0][7:])))
            print('.',end='')
        #print("train data file", os.getcwd() + txt[i][0][7:])
        #print("label data raw", cv2.imread(os.getcwd() + '/CamVid/trainannot/0001TP_006690.png'))
        return np.array(data), np.array(label)

    def preprocess_inputs(self, X):
    ### @ https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py
        """Preprocesses a tensor encoding a batch of images.
        # Arguments
            x: input Numpy tensor, 4D.
            data_format: data format of the image tensor.
            mode: One of "caffe", "tf".
                - caffe: will convert the images from RGB to BGR,
                    then will zero-center each color channel with
                    respect to the ImageNet dataset,
                    without scaling.
                - tf: will scale pixels between -1 and 1,
                    sample-wise.
        # Returns
            Preprocessed tensor.
        """
        return imagenet_utils.preprocess_input(X)

    def reshape_labels(self, y):
        return np.reshape(y, (len(y), self.data_shape, self.classes))
