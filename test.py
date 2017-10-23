# coding=utf-8
from mxnet import ndarray as nd
from mxnet import autograd
import random

num_inputs = 2
num_examples = 100
true_w = [2, -3.4]
true_b = 4.2

x = nd.arange(101, 201).reshape((20, 5))

batch_size = 10
idx = list(range(num_examples))
random.shuffle(idx)
# for i in range(0, num_examples, batch_size):
#     j = nd.array(idx[i:min(i + batch_size, num_examples)])
#     print(j, nd.take(x, j))

print(nd.take(x, nd.array([1, 2])))
