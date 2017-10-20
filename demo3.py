from mxnet import ndarray as nd
import numpy as np
x = np.ones((2, 3))
y = nd.array(x)
z = y.asnumpy()
#print x, y, z

a = nd.arange(3).reshape((3, 1))
b = nd.arange(2).reshape((1, 2))
print ('a:', a)
print ('b:', b)
print ('a+b:', a+b)

