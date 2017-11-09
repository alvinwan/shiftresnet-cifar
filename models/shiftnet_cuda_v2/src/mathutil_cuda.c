#include <THC/THC.h>
#include "mathutil_cuda_kernel.h"

#include <stdio.h>

extern THCState *state;

int broadcast_sum(THCudaTensor *a_tensor, THCudaTensor *b_tensor, int x, int y)
{
    float *a = THCudaTensor_data(state, a_tensor);
    float *b = THCudaTensor_data(state, b_tensor);
    cudaStream_t stream = THCState_getCurrentStream(state);

    broadcast_sum_cuda(a, b, x, y, stream);

    return 1;
}

int inspect(THCudaTensor *a_tensor)
{
    float *a = THCudaTensor_data(state, a_tensor);
    cudaStream_t stream = THCState_getCurrentStream(state);

    //broadcast_sum_cuda(a, b, x, y, stream);

    int a_ndim = THCudaTensor_nDimension(state, a_tensor);
    for (int dim = 0; dim < a_ndim; ++dim) {
        long size = THCudaTensor_size(state, a_tensor, dim);
        long stride = THCudaTensor_stride(state, a_tensor, dim);
        printf("DEBUG: inspect: size[%d]: %ld stride[%d]: %ld\n",
            dim, size, dim, stride);
    }

    return 1;
}
