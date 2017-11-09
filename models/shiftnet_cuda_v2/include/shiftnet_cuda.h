int moduloshift3x3_nchw(THCudaTensor *src_tensor, THCudaTensor *dst_tensor);
int moduloshift3x3bwd_nchw(THCudaTensor *src_tensor, THCudaTensor *dst_tensor);
int moduloshiftgeneric_nchw(THCudaTensor *src_tensor, THCudaTensor *dst_tensor, int kernel_size, int dilate_factor, int direction);
