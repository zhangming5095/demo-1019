import OpenSSL.SSL
from mxnet import ndarray as nd
from mxnet import autograd as ag
import random

# 数据集
# y[i] = 3 * X[i][0] - 4 * X[i][1] + 5 + noise
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2

x = nd.random_normal(shape=(num_examples, num_inputs))
#y = nd.dot(x, nd.array(true_w).T) + true_b # 同下
y = true_w[0] * x[:, 0] + true_w[1] * x[:, 1] + true_b
y += .01 * nd.random_normal(shape=y.shape)

# 数据读取
batch_size = 10


def data_iter():
    idx = list(range(num_examples))
    random.shuffle(idx)
    for i in range(0, num_examples, batch_size):
        j = nd.array(idx[i:min(i + batch_size, num_examples)])
        yield nd.take(x, j), nd.take(y, j)


# 初始化模型参数
# w = nd.random_normal(shape=(1, num_inputs)) # 同下,对应 nd.dot(X, w.T) + b
w = nd.random_normal(shape=(num_inputs, 1))
b = nd.zeros((1,))
params = [w, b]
# 之后训练时我们需要对这些参数求导来更新它们的值，所以我们需要创建它们的梯度
for param in params:
    param.attach_grad()


# 定义模型
def net(X):
    return nd.dot(X, w) + b


# 损失函数
def square_loss(yhat, y):
   # print(yhat, y)
    return (yhat - y.reshape(yhat.shape)) ** 2


# 优化:随机梯度下降,每一步，我们将模型参数沿着梯度的反方向走特定距离，这个距离一般叫学习率
def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

# 训练
epochs = 5
learning_rate = .001
for e in range(epochs):
    total_loss = 0
    for data, label in data_iter():
        with ag.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        SGD(params, learning_rate)

        total_loss += nd.sum(loss).asscalar()
    print('Epoch %d, average loss: %f' % (e, total_loss / num_examples))

print(w, true_w)
print(b, true_b)
