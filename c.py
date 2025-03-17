from tinygrad import Tensor
from tinygrad.helpers import make_tuple, prod, ceildiv
from tinygrad.dtype import dtypes
from itertools import product
'''
todo
- get my pool to work with 1d tensors
- find more robust ways to test _pool and associated operations
    - existing tests are good
    - how to test performance and various enviroments (some ci?)


 - tuple strides (if tuple, turn into tensor and do a broadcasted add)
 - tuple dilation
 - padding 
 - ceil mode

options
1. replace _pool with indexed pool
    - hardest but best. need to see if performance improves.
2. add return_indicies to _pool
2. add return_indicies to max_pool2d
'''

### Params

width=4
dims=3
kernel_size = 2
stride = 1
dilation = 1
padding = 0
ceil_mode = False

shape = tuple([width]*dims)
#a = Tensor.arange(0,width ** dims).reshape(*shape)
a = Tensor.randint(prod(shape)).reshape(shape)

### Manual maxpool
'''
print(a.numpy())
m = a.max_pool2d(kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding, ceil_mode=ceil_mode)
'''


pads = a._resolve_pool_pads(padding, len(k_ := make_tuple(kernel_size, 2))) 
if ceil_mode: pads = a._apply_ceil_mode(pads, k_, stride if stride is not None else k_, dilation)
pad_actual = a.pad(pads, value=dtypes.min(a.dtype))
pools = pad_actual._pool(k_, stride if stride is not None else k_, dilation)
maxpool = pools.max(tuple(range(-len(k_), 0)))

d_, s_ = make_tuple(dilation, len(k_)), make_tuple(stride, len(k_))
i_ = a.shape[-len(k_):]

o_ = [ceildiv(i - d*(k-1), s) for i, d, k, s in zip(i_, d_, k_, s_)]
pool_shape = [shape[d] for d in range(dims-2,0,-1)] + o_ + list(k_)

### Idxs of pooling operation

window_samples = Tensor(list(product(*[[k * dilation for k in range(k_i)] for k_i in k_]))) # relative sampling indices within a window
k1, k2 = pool_shape[-4], pool_shape[-3] # number of pools in each direction
x, y = Tensor.meshgrid(Tensor.arange(k1), Tensor.arange(k2))
pool_indices = Tensor.stack(x.flatten(), y.flatten()).transpose() * stride # index of top-left corner of each pool
idxs = window_samples.reshape(1,-1,len(k_)) + pool_indices.reshape(-1,1,len(k_)) 
idxs = (idxs * Tensor([shape[-1],1])).sum(-1) # the idxs for a single 2d matrix with these settings
idxs = idxs.repeat(prod(shape[:-len(k_)]), 1, 1) # replicate to each dimension

idxs = idxs.reshape(pool_shape)
my_pools = a.flatten()[idxs]

print("a:")
print(a.numpy())

print("Pools:")
print(pools.numpy())

print("Idxs:")
print(idxs.numpy())

print("Maxpool:")
print(maxpool.numpy())

# Flatten the window dimensions in both `pools` and `idxs`
flat_pools = pools.reshape(*pools.shape[:-len(k_)], prod(k_))
flat_idxs  = idxs.reshape(*idxs.shape[:-len(k_)], prod(k_))
maxpool_flat_idx = flat_idxs.gather(dim=-1, index=flat_pools.argmax(-1).unsqueeze(-1)).squeeze(-1)

# maxpool_flat_idx now holds the flattened indices for the max element in each pooling window.
print("Maxpool indices:")
print(maxpool_flat_idx.numpy())

import torch
b = torch.tensor(a.numpy()).unsqueeze(0)
out, ids = torch.nn.functional.max_pool2d(b, kernel_size=kernel_size, dilation=dilation, stride=stride, 
                            return_indices=True, padding=padding)

print("TORCH")
print(ids)
print((ids.numpy() == maxpool_flat_idx.numpy()).all())


