# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 23:40:40 2017

@author: nadha
"""
import os 

#import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from scipy import io as sio
from skimage.feature import hog
from skimage.transform import resize
#from skimage import filters
from skimage import color


def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


data = np.zeros((50000,3072))
labels = np.zeros((50000))
test_data = np.zeros((10000,3072))
test_labels =np.zeros((10000))
for i in range(0,5):
    A = unpickle('data_batch_'+str(i+1))
    data[i*10000:(i+1)*10000,:] = (A['data'])
    labels[i*10000:(i+1)*10000] = A['labels']

A = unpickle('data_batch_'+str(i+1))
test_data[0:10000,:] = (A['data'])
test_labels[0:10000] = A['labels']

f_set = []
for i in range(0,50000):
    imag1 = np.reshape(data[i,:],(32,32,3))
    image = color.rgb2gray(imag1)
    image = resize(image, (32, 32), mode='reflect')
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(12, 12),
                        cells_per_block=(1, 1), visualise=True)
    f_set.append(fd)

arr = np.array(f_set)
#print arr
X = {'X':arr}
sio.savemat('X.mat',X)
lb = {'Y':labels}
sio.savemat('Y.mat',lb)


f_set = []
for i in range(0,10000):
    imag1 = np.reshape(test_data[i,:],(32,32,3))
    image = color.rgb2gray(imag1)
    image = resize(image, (32, 32), mode='reflect')
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualise=True)
    f_set.append(fd)

arr = np.array(f_set)
X = {'X_test':arr}
sio.savemat('X_test.mat',X)
lb = {'Y_test':labels}
sio.savemat('Y_test.mat',lb)