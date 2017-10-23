# coding=utf-8
from mxnet import gluon
from mxnet import ndarray as nd


# 获取数据
def transform(data, label):
    return data.astype('float32') / 255, label.astype('float32')


mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
# mnist_test = gluon.data.vision.FashionMNIST(train=False, transform=transform)
print('你好')
