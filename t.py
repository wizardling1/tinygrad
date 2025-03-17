from tinygrad import Tensor
from tinygrad.helpers import make_tuple, prod
from tinygrad.dtype import dtypes
from itertools import product

'''
my options
- use the pythonic code only to return indicies for maxpool2d
- use pythonic code only to return indicies for _pool; argmax for maxpool2d (makes more sense)
- replace _pool with pythonic code (need to test speed)
- move pythonic code into tensor code; create a tensor to generate indicies of pooling; proceed
    with other methods

what i should do
- calculate the relative amount of computation occuring by this index calculation
- create a tensor based version and compare speed
- compare tensor based index computation vs current pool.
'''

### Params

width=2
dims=3
kernel_size = 2
stride = 1
dilation = 1
padding = 0
ceil_mode = False

a = Tensor.arange(0,width ** dims).reshape(*([width]*dims))

### Manual maxpool

shape = a.shape
pads = a._resolve_pool_pads(padding, len(k_ := make_tuple(kernel_size, 2))) 
if ceil_mode: pads = a._apply_ceil_mode(pads, k_, stride if stride is not None else k_, dilation)
pad_actual = a.pad(pads, value=dtypes.min(a.dtype))
pools = pad_actual._pool(k_, stride if stride is not None else k_, dilation)
maxpool = pools.max(tuple(range(-len(k_), 0)))

#### Return_indices
# create indicies for first 2d matrix
window_samples = Tensor(list(product(*[[k * dilation for k in range(k_i)] for k_i in k_])))
ranges = [Tensor.arange(s) for s in pools.shape[:-2]][::-1]
pool_indices = ranges[0]
r = ranges[1]
repeat = pool_indices.reshape(1,1,-1).repeat((1,len(r),1)).reshape(1,-1)
inter = r.repeat_interleave(pool_indices.shape[-1])
pool_indices = inter.stack(*repeat)

'''
for i, r in enumerate(ranges[1:]):
    print(r.numpy())
    repeat = pool_indices.reshape(i+1,1,-1).repeat((1,len(r),1)).reshape(i+1,-1)
    if i == 0:
        inter = r.repeat_interleave(pool_indices.shape[-1])
        pool_indices = inter.stack(*repeat)
    else:
        pool_indices = repeat
'''

# todo: make with with tuple strides (if tuple, turn into tensor and do a broadcasted add)
# todo: make idxs work with any dims
pool_indices = pool_indices.transpose() * stride
idxs = window_samples.reshape(1,-1,2) + pool_indices.reshape(-1,1,2)
# flatten and reshape
#strides = Tensor([int(Tensor(shape[i+1:]).prod().numpy()) for i in range(len(shape))])  # Compute strides
strides = Tensor([shape[-1], 1])
idxs = (idxs * strides).sum(-1).reshape(1,*pools.shape[dims-2:])

# expand to each dimension
nelem = prod(shape[-2:])
extra_dims = []
for s in range(prod(shape[:-2])-1):
    extra_dims.append(idxs + (nelem * (s + 1)))
idxs = idxs.stack(*extra_dims).reshape(pools.shape)

print("a:")
print(a.numpy())

print("Pools:")
print(pools.numpy())

print("Idxs:")
print(idxs.numpy())


''' test code
print("Maxpools:")
print(maxpool.numpy())

# extract argmax 
argmax = pools.argmax(0)
print(argmax.shape)
print(k_)
print(argmax.numpy())
argmax = Tensor.arange(prod(k_)).stack(*argmax.reshape(dims,-1)).transpose()
print("Argmax of pools:")
#print(argmax.numpy())

print("out:")
maxpool_idxs = [idxs[*t] for t in [*argmax]]
maxpool_idxs = maxpool_idxs[0].stack(*maxpool_idxs[1:]).reshape(maxpool.shape)
print(maxpool_idxs.numpy())

### Old pythonic code

# get pool indicies, python version
pool_indices_py = list(product(*(range(s) for s in pools.shape[:-2])))
tuple_add = lambda t1, t2: tuple(a+b for a, b in zip(t1, t2))
pool_start_locs = [tuple(i * stride for i in p_idx) for p_idx in pool_indices]
idxs = [[tuple_add(sample, loc) for sample in window_samples] for loc in pool_start_locs]
print((pool_indices.numpy() == np.array(pool_indices_py)).all())

print("Pools experiments")
k_eff = tuple(map(lambda x: (x - 1) * dilation + 1, k_)) # effective kernel size with dilation
s_ = make_tuple(stride, 2)

padded_pools = []

for i in range(pools.shape[0]):
    for j in range(pools.shape[1]):
        start_row = i * s_[0] 
        start_col = j * s_[1]
        p = pools[i,j]

        pads = [start_col, shape[0]-(start_col + k_eff[0]), start_row, shape[1]-(start_row + k_eff[1])]

        pad_actual = p.pad(pads, value=dtypes.min(p.dtype))
        print(pad_actual.numpy())
        print(".")

        padded_pools.append(pad_actual.flatten())

print("Final")
final = padded_pools[0].stack(*padded_pools[1:]).reshape(pools.shape[0], pools.shape[1],-1).argmax(-1) 
print(final.numpy())
'''
