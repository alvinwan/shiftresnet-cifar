# https://gist.github.com/tonyseek/7821993
import torch
from torch.utils.ffi import create_extension
import glob
import os

abs_path = os.path.dirname(os.path.realpath(__file__))
extra_objects = [os.path.join(abs_path, 'build/shiftnet_cuda_kernels.so')]
#extra_objects += glob.glob('/usr/local/cuda-8.0/lib64/*.a')

ffi = create_extension(
    'shiftnet_cuda',
    headers=['include/shiftnet_cuda.h'],
    sources=['src/shiftnet_cuda.c'],
    define_macros=[('WITH_CUDA', None)],
    relative_to=__file__,
    with_cuda=True,
    extra_objects=extra_objects,
    include_dirs=[os.path.join(abs_path, 'include')]
)

if __name__ == '__main__':
    assert torch.cuda.is_available(), 'Please install CUDA for GPU support.'
    ffi.build()
