from tinygrad import Tensor

a = Tensor.empty(4)
a = a + 1
a.realize()
