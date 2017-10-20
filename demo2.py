# coding=utf-8
from mxnet import ndarray as nd


x = nd.ones((3, 4))  # 都是1
y = nd.random_normal(0, 1, shape=(3, 4))  # 服从正态分布
# print y
a = nd.array([[1, 1],
              [0, 1]])
b = nd.array([[1, 0],
              [1, 1]])
# print x, y, a, b
# print a + b
# print nd.exp(b)
# print nd.dot(a, b.T)  # 矩阵乘法

# before = id(y)
# print before
# y = y + x
# print id(y) == before

z = nd.zeros_like(x)  # 全为0, 形状像 x
print z
before = id(z)
z[:] = x + y
print id(z) == before

