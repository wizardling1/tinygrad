from tinygrad import Tensor, dtypes
import numpy as np

a = Tensor([1,2,1,4,5,6,7,8,9,10])
b = Tensor([1,2,3,4,5,6,7,8,9,10])

c = Tensor.randint(10)

print(c.numpy())
print(c.topk2(5))

