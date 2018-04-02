

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
    image = resize(image, (16, 16), mode='reflect')
    
    f_set.append(image[:])

arr = np.array(f_set)
#print arr
X = {'X':arr}
sio.savemat('X_image.mat',X)
lb = {'Y':labels}
sio.savemat('Y_image.mat',lb)


f_set = []
for i in range(0,10000):
    imag1 = np.reshape(test_data[i,:],(32,32,3))
    image = color.rgb2gray(imag1)
    image = resize(image, (16, 16), mode='reflect')
    
    
    f_set.append(image[:])
arr = np.array(f_set)
X = {'X_test':arr}
sio.savemat('X_image_test.mat',X)
lb = {'Y_test':test_labels}
sio.savemat('Y_image_test.mat',lb)
