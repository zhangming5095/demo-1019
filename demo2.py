# coding=utf-8
from mxnet import ndarray as nd

x = nd.ones((3, 4))
y = nd.random_normal(0, 1, shape=(3, 4))
a = nd.array([[1, 1],
              [0, 1]])
b = nd.array([[1, 0],
              [1, 1]])
# print x, y, a, b
# print a + b
# print nd.exp(b)
# print nd.dot(a, b.T)

# before = id(y)
# print before
# y = y + x
# print id(y) == before

z = nd.zeros_like(x)
before = id(z)
z[:] = x + y
print id(z) == before

