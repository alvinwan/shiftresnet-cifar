#include <cuda_runtime.h>

void shiftnet_cuda_moduloshift3x3_nchw_float32(
    float *src,
    float *dst,
    int batch_sz,
    int channels,
    int height,
    int width,
    cudaStream_t stream);

void shiftnet_cuda_moduloshift3x3bwd_nchw_float32(
    float *src,
    float *dst,
    int batch_sz,
    int channels,
    int height,
    int width,
    cudaStream_t stream);

void shiftnet_cuda_moduloshiftgeneric_nchw_float32(
    float *src,
    float *dst,
    int batch_sz,
    int channels,
    int height,
    int width,
    int kernel_size,
    int dilate_factor,
    int direction,
    cudaStream_t stream);
