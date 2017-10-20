# coding=utf-8
import mxnet.ndarray as nd
import mxnet.autograd as ag

def f(a):
    b = a * 2
    while nd.norm(b).asscalar() < 1000:
        b = b * 2
    if nd.sum(b).asscalar() > 0:
        c = a
    else:
        c = 100 * b
    return c

a = nd.random_normal(shape = 3)
a.attach_grad()
with ag.record():
    c = f(a)
c.backward()
print a.grad == c/a

