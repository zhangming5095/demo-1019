# coding=utf-8
from mxnet import ndarray as nd
from mxnet import autograd

a = nd.array([[1, 1], [2, 2]])
b = nd.arange(3)
c = nd.norm(a)  # 矩阵a 的所有值的平方和 的算术平方根,结果是一个 矩阵
print a, c
print type(nd.sum(b))
print c.asscalar()
