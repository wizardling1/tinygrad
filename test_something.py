import numpy as np
import torch
import torch.nn.functional as F
import pytest
from tinygrad import Tensor  # adjust this import based on your tinygrad setup

# --- Helper Functions ---

def run_avg_pool2d_tinygrad(x_np, kernel_size=(2, 2), stride=None, dilation=1, padding=0,
                            ceil_mode=False, count_include_pad=True):
    """Runs tinygrad's avg_pool2d and returns a NumPy array."""
    x = Tensor(x_np)
    y = x.avg_pool2d(kernel_size, stride, dilation, padding, ceil_mode, count_include_pad)
    return y.numpy()

def run_avg_pool2d_torch(x_np, kernel_size=(2, 2), stride=None, dilation=1, padding=0,
                         ceil_mode=False, count_include_pad=True):
    """Runs PyTorch's avg_pool2d and returns a NumPy array."""
    x = torch.tensor(x_np, dtype=torch.float32)
    if stride is None:
        stride = kernel_size
    y = F.avg_pool2d(x, kernel_size, stride, padding, ceil_mode, count_include_pad)
    return y.detach().cpu().numpy()

def run_max_pool2d_tinygrad(x_np, kernel_size=(2, 2), stride=None, dilation=1, padding=0,
                            ceil_mode=False):
    """Runs tinygrad's max_pool2d and returns a NumPy array."""
    x = Tensor(x_np)
    y = x.max_pool2d(kernel_size, stride, dilation, padding, ceil_mode)
    return y.numpy()

def run_max_pool2d_torch(x_np, kernel_size=(2, 2), stride=None, dilation=1, padding=0,
                         ceil_mode=False):
    """Runs PyTorch's max_pool2d and returns a NumPy array."""
    x = torch.tensor(x_np, dtype=torch.float32)
    if stride is None:
        stride = kernel_size
    y = F.max_pool2d(x, kernel_size, stride, padding, dilation, ceil_mode)
    return y.detach().cpu().numpy()

# --- Fixtures ---

@pytest.fixture
def random_input():
    """Creates a reproducible random input tensor."""
    np.random.seed(0)
    return np.random.rand(1, 3, 5000, 5000).astype(np.float32)

# --- Accuracy Tests ---

def test_avg_pool2d_accuracy(random_input):
    kernel_size = (2, 2)
    stride = 2
    padding = 0
    dilation = 1
    ceil_mode = False
    count_include_pad = True

    y_tiny = run_avg_pool2d_tinygrad(random_input, kernel_size, stride, dilation, padding, ceil_mode, count_include_pad)
    #print("y_tiny"); print(y_tiny)
    y_torch = run_avg_pool2d_torch(random_input, kernel_size, stride, dilation, padding, ceil_mode, count_include_pad)
    #print("y_torch"); print(y_torch)
    np.testing.assert_allclose(y_tiny, y_torch, atol=1e-5)

def test_max_pool2d_accuracy(random_input):
    kernel_size = (2, 2)
    stride = 2
    padding = 0
    dilation = 1
    ceil_mode = False

    y_tiny = run_max_pool2d_tinygrad(random_input, kernel_size, stride, dilation, padding, ceil_mode)
    y_torch = run_max_pool2d_torch(random_input, kernel_size, stride, dilation, padding, ceil_mode)
    np.testing.assert_allclose(y_tiny, y_torch, atol=1e-5)

# Separate tests for each implementation so that the benchmark fixture is only used once per test.

def test_avg_pool2d_tinygrad_performance(benchmark, random_input):
    kernel_size = (2, 2)
    stride = 2
    padding = 0
    dilation = 1
    ceil_mode = False
    count_include_pad = True

    result = benchmark(lambda: run_avg_pool2d_tinygrad(random_input, kernel_size, stride, dilation, padding, ceil_mode, count_include_pad))
    print("tinygrad avg_pool2d time:", result)

def test_avg_pool2d_torch_performance(benchmark, random_input):
    kernel_size = (2, 2)
    stride = 2
    padding = 0
    dilation = 1
    ceil_mode = False
    count_include_pad = True

    result = benchmark(lambda: run_avg_pool2d_torch(random_input, kernel_size, stride, dilation, padding, ceil_mode, count_include_pad))
    print("torch avg_pool2d time:", result)

def test_max_pool2d_tinygrad_performance(benchmark, random_input):
    kernel_size = (2, 2)
    stride = 2
    padding = 0
    dilation = 1
    ceil_mode = False

    result = benchmark(lambda: run_max_pool2d_tinygrad(random_input, kernel_size, stride, dilation, padding, ceil_mode))
    print("tinygrad max_pool2d time:", result)

def test_max_pool2d_torch_performance(benchmark, random_input):
    kernel_size = (2, 2)
    stride = 2
    padding = 0
    dilation = 1
    ceil_mode = False

    result = benchmark(lambda: run_max_pool2d_torch(random_input, kernel_size, stride, dilation, padding, ceil_mode))
    print("torch max_pool2d time:", result)

