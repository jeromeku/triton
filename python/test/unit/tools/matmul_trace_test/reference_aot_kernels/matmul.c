#include <cuda.h>
#include <stdint.h>
#include <assert.h>

// launcher for: matmul_16x16x16_warps1xstages3
CUresult matmul_bdd999c7_0d1d2d3d4d5d6d7c8d9c10d11c(CUstream stream, CUdeviceptr C, CUdeviceptr A, CUdeviceptr B, int32_t M, int32_t N, int32_t K, int32_t stride_cm, int32_t stride_am, int32_t stride_bk);
CUresult matmul_3db33494_01234567891011(CUstream stream, CUdeviceptr C, CUdeviceptr A, CUdeviceptr B, int32_t M, int32_t N, int32_t K, int32_t stride_cm, int32_t stride_cn, int32_t stride_am, int32_t stride_ak, int32_t stride_bk, int32_t stride_bn);

CUresult matmul_16x16x16_warps1xstages3(CUstream stream, CUdeviceptr C, CUdeviceptr A, CUdeviceptr B, int32_t M, int32_t N, int32_t K, int32_t stride_cm, int32_t stride_cn, int32_t stride_am, int32_t stride_ak, int32_t stride_bk, int32_t stride_bn){
  if ((C % 16 == 0) && (A % 16 == 0) && (B % 16 == 0) && (M % 16 == 0) && (N % 16 == 0) && (K % 16 == 0) && (stride_cm % 16 == 0) && (stride_cn == 1) && (stride_am % 16 == 0) && (stride_ak == 1) && (stride_bk % 16 == 0) && (stride_bn == 1))
    return matmul_bdd999c7_0d1d2d3d4d5d6d7c8d9c10d11c(stream, C, A, B, M, N, K, stride_cm, stride_am, stride_bk);
if (1)
    return matmul_3db33494_01234567891011(stream, C, A, B, M, N, K, stride_cm, stride_cn, stride_am, stride_ak, stride_bk, stride_bn);

  return CUDA_ERROR_INVALID_VALUE;
}

// load for: matmul_16x16x16_warps1xstages3
void load_matmul_bdd999c7_0d1d2d3d4d5d6d7c8d9c10d11c();
void load_matmul_3db33494_01234567891011();
void load_matmul_16x16x16_warps1xstages3() {
  load_matmul_bdd999c7_0d1d2d3d4d5d6d7c8d9c10d11c();
  load_matmul_3db33494_01234567891011();
}

// unload for: matmul_16x16x16_warps1xstages3
void unload_matmul_bdd999c7_0d1d2d3d4d5d6d7c8d9c10d11c();
void unload_matmul_3db33494_01234567891011();
void unload_matmul_16x16x16_warps1xstages3() {
  unload_matmul_bdd999c7_0d1d2d3d4d5d6d7c8d9c10d11c();
  unload_matmul_3db33494_01234567891011();
}

typedef CUresult (*kernel_func_t)(CUstream stream, CUdeviceptr C, CUdeviceptr A, CUdeviceptr B, int32_t M, int32_t N, int32_t K, int32_t stride_cm, int32_t stride_cn, int32_t stride_am, int32_t stride_ak, int32_t stride_bk, int32_t stride_bn);
kernel_func_t matmul_kernels[] = {
  matmul_16x16x16_warps1xstages3,
};

int matmul_get_num_algos(void){
  return (int)sizeof(matmul_kernels);
}

CUresult matmul(CUstream stream, CUdeviceptr C, CUdeviceptr A, CUdeviceptr B, int32_t M, int32_t N, int32_t K, int32_t stride_cm, int32_t stride_cn, int32_t stride_am, int32_t stride_ak, int32_t stride_bk, int32_t stride_bn, int algo_id){
  assert (algo_id < (int)sizeof(matmul_kernels));
  return matmul_kernels[algo_id](stream, C, A, B, M, N, K, stride_cm, stride_cn, stride_am, stride_ak, stride_bk, stride_bn);
}

void load_matmul(void){
  load_matmul_16x16x16_warps1xstages3();
}

void unload_matmul(void){
  unload_matmul_16x16x16_warps1xstages3();
}


CUresult matmul_default(CUstream stream, CUdeviceptr C, CUdeviceptr A, CUdeviceptr B, int32_t M, int32_t N, int32_t K, int32_t stride_cm, int32_t stride_cn, int32_t stride_am, int32_t stride_ak, int32_t stride_bk, int32_t stride_bn){
  return matmul(stream, C, A, B, M, N, K, stride_cm, stride_cn, stride_am, stride_ak, stride_bk, stride_bn, 0);
}
