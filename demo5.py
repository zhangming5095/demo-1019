# coding=utf-8
import mxnet.ndarray as nd
import mxnet.autograd as ag


# 对控制流求导
def f(a):
    b = a * 2
    while nd.norm(b).asscalar() < 1000:
        b = b * 2
    if nd.sum(b).asscalar() > 0:
        c = a
    else:
        c = 100 * b
    return c


a = nd.random_normal(shape=3)
# 通过 NDArray 的⽅法 attach_grad()来要求系统申请对应的空间
a.attach_grad()
# 要使⽤ autograd⾥的 record() 函数来显式的要求 MXNet 记录我们需要求导的程序
with ag.record():
    c = f(a)
c.backward()  # 通过 z.backward() 来进⾏求导
print(a.grad == c / a)
