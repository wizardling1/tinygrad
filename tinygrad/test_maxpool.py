from tinygrad import Tensor
from tinygrad import dtypes
from tinygrad.helpers import make_tuple

a = Tensor([
    [1,1,1,1,1],
    [2,2,2,2,2],
    [3,3,3,3,3],
    [3,3,3,3,3],
    [3,3,3,3,3],
    ])
b, indices = a.max_pool2d(kernel_size=(2,2), stride=2, 
                        return_indices=True)

print(b.numpy())

'''
kernel_size = (2,2)
padding = 0
stride=1
pads = a._resolve_pool_pads(padding,len(kernel_size))

# step1 seems to be to apply padding? yes.
step1 = a.pad(pads, value=dtypes.min(a.dtype))
# step2 is to generate all the pools
step2 = step1._pool(kernel_size, stride, 1)
# step3 is to take the max along some sort of axis..
neg_len = -len(kernel_size)
max_axis = tuple(range(neg_len,0))
print(max_axis)
step3 = step2.max(max_axis)

print(step1.numpy())

print(step2.numpy().shape)

print(step3.numpy())
'''

