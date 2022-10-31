import os
import struct
import numpy as np
import matplotlib.pyplot as plt

data_path = r'.\\dataset\\minist\\'

def minst_load(path, kind='train'):
    """Load MNIST data from `path`"""
    images_path = os.path.join(path, '{}-images.idx3-ubyte'.format(kind))
    labels_path = os.path.join(path, '{}-labels.idx1-ubyte'.format(kind))

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels
    # images: n x m, n is number of samples, m is 784 pixels
    # labels: n
features, labels = minst_load(data_path)  # features: (60000, 784), labels: (60000,)
# np.savetxt('train_img.csv', features, fmt='%i', delimiter=',')     convert numpy array to csv file
# features = np.genfromtxt('train_img.csv', dtype=int, delimiter=',')   read csv file into numpy array

# Show
x_train, y_train = [], labels
for i in range(len(features)):
    x_train.append(np.reshape(features[i], (28, 28)))
x_train = np.array(x_train)

plt.imshow(x_train[111], cmap='gray')  # Grayscale
plt.show()
print(y_train[111])
