import OpenSSL.SSL
from mxnet import ndarray as nd
from mxnet import autograd as ag
from mxnet import gluon

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2

x = nd.random_normal(shape=(num_examples, num_inputs))
y = x[:, 0] * true_w[0] + x[:, 1] * true_w[1] + true_b
y += .01 * nd.random_normal(shape=y.shape)

batch_size = 100
dataset = gluon.data.ArrayDataset(x, y)
data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True)

# 首先我们定义一个空的模型
net = gluon.nn.Sequential()
# 然后我们加入一个Dense层，它唯一必须要定义的参数就是输出节点的个数，在线性模型里面是1
net.add(gluon.nn.Dense(1))

# 初始化模型参数,默认随机初始化方法
net.initialize()

# 损失函数
square_loss = gluon.loss.L2Loss()

# 优化,无需手动实现随机梯度下降，我们可以用创建一个Trainer的实例，并且将模型参数传递给它就行
trainer = gluon.Trainer(
   net.collect_params(), 'sgd', {'learning_rate': 0.001})

# 训练
epochs = 5
for e in range(epochs):
    total_loss = 0
    for data, label in data_iter:
        with ag.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        trainer.step(batch_size)
        total_loss += nd.sum(loss).asscalar()
    print("Epoch %d, average loss: %f" % (e, total_loss / num_examples))



