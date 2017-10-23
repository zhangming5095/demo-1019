# coding=utf-8
import mxnet.ndarray as nd
import mxnet.autograd as ag

x = nd.array([[1, 2],
              [3, 4]])

# 当进⾏求导的时候，我们需要⼀个地⽅来存 x 的导数，
# 这个可以通过 NDArray 的⽅法 attach_grad()来要求系统申请对应的空间
x.attach_grad()

# 默认条件下，MXNet 不会⾃动记录和构建⽤于求导的计算图，
# 我们需要使⽤ autograd ⾥的 record() 函数来显式的要求 MXNet 记录我们需要求导的程序

with ag.record():
    y = x * 2
    z = y * x

# 通过 z.backward() 来进⾏求导
# 如果 z 不是⼀个标量，那么 z.backward()等价于 nd.sum(z).backward().
# y.backward()
# print(x.grad)

z.backward()
print(x.grad)
