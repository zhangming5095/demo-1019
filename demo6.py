# coding=utf-8
# 线性回归
# y = X * [2, -1].T + 1 + noise ,这⾥噪⾳服从均值 0 和标准差为 0.01 的正态分布

import random

from mxnet import ndarray as nd
from mxnet import autograd

# 1,创建数据集
num_inputs = 2
num_examples = 10000
true_w = [2, -1]
true_b = 1
X = nd.random_normal(shape=(num_examples, num_inputs))  # 服从正态分布
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b  # = nd.dot(X,nd.array(true_w).T) + true_b
y += .01 * nd.random_normal(shape=y.shape)

# 2,数据读取
# 定义⼀个函数它每次返回 batch_size 个随机的样本和对应的⽬标
# 通过 python 的 yield 来构造⼀个迭代器
batch_size = 10


def data_iter():
    idx = list(range(num_examples))  # 产生一个从0 到 num_examples(不含) 的有序集合
    random.shuffle(idx)  # 打乱顺序
    for i in range(0, num_examples, batch_size):
        j = nd.array(idx[i:min(i + batch_size, num_examples)])
        yield nd.take(X, j), nd.take(y, j)


# 下⾯代码读取第⼀个随机数据块
#  for data, label in data_iter():
#      print data, label
#      break


# 随机初始化模型参数
w = nd.random_normal(shape=(num_inputs, 1))
b = nd.zeros((1,))
params = [w, b]

# 之后训练时我们需要对这些参数求导来更新它们的值，所以我们需要创建它们的梯度
for param in params:
    param.attach_grad()


# 定义模型
def net(X):
    return nd.dot(X, w) + b


# 损失函数,差的平方
def square_loss(yhat, y):
    # 注意这⾥我们把 y 变形成 yhat 的形状来避免⾃动⼴播
    return (yhat - y.reshape(yhat.shape)) ** 2


# 优化
def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad


# 训练
epochs = 5
learning_rate = .001
for e in range(epochs):
    total_loss = 0  # 差方和
    for data, label in data_iter():  # 数据集
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)  # 差方
        loss.backward()
        SGD(params, learning_rate)

        total_loss += nd.sum(loss).asscalar()
    print('Epoch %d, average loss: %f' % (e, total_loss / num_examples))
