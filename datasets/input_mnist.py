from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os


ROOTDIR = '/home/the/Datasets/MNIST/'

# 载入MNIST数据集，如果指定地址/path/to/MNIST_data下没有已经下载好的数据集，
# 那么tensorflow自动从上表给出的网站下载数据。
mnist = input_data.read_data_sets(ROOTDIR, one_hot=True)

# 打印Training data size:  55000。
print("Training data size: ", mnist.train.num_examples)

# 打印Validation data size:  5000。
print("Validating data size: ", mnist.validation.num_examples)

# 打印Testing data size:  10000。
print("Testing data size: ", mnist.test.num_examples)

# 打印Example training data: [0.  0.  0.  ...  0.380  0.376  ...  0.]。
# print("Example training data: ", mnist.train.images[0])



# 打印Example training data label:
# [0.  0.  0.  0.  0.  0.  0.  1.  0.  0.]
# print("Example training data label: ", mnist.train.labels[0])
train_data = mnist.train.images
train_data_mean = np.mean(train_data, axis=0)
print(train_data_mean.shape)
# np.save(train_data_mean, './mnist_mean.npy', allow_pickle=False)
# with open(os.path.join(ROOTDIR, 'train-images-idx3-ubyte.gz')) as f:
#     loaded = np.fromfile(file=f, dtype=np.uint8)
#     train_data = loaded[16:].reshape((60000, 784))