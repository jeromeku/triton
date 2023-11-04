#ifndef TT_KERNEL_INCLUDES
#define TT_KERNEL_INCLUDES

#include <cuda.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#endif

void unload_add_kernel_8d4b99fa_0d1d2d3de(void);
void load_add_kernel_8d4b99fa_0d1d2d3de(void);
// tt-linker: add_kernel_8d4b99fa_0d1d2d3de:CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements:1024_warps4xstages3
CUresult add_kernel_8d4b99fa_0d1d2d3de(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements);
