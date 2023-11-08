#include <cuda.h>

CUresult matmul_16x16x16_warps1xstages3(CUstream stream, CUdeviceptr C, CUdeviceptr A, CUdeviceptr B, int32_t M, int32_t N, int32_t K, int32_t stride_cm, int32_t stride_cn, int32_t stride_am, int32_t stride_ak, int32_t stride_bk, int32_t stride_bn);
void load_matmul_16x16x16_warps1xstages3();
void unload_matmul_16x16x16_warps1xstages3();
    
int matmul_get_num_algos(void);

CUresult matmul_default(CUstream stream, CUdeviceptr C, CUdeviceptr A, CUdeviceptr B, int32_t M, int32_t N, int32_t K, int32_t stride_cm, int32_t stride_cn, int32_t stride_am, int32_t stride_ak, int32_t stride_bk, int32_t stride_bn);
CUresult matmul(CUstream stream, CUdeviceptr C, CUdeviceptr A, CUdeviceptr B, int32_t M, int32_t N, int32_t K, int32_t stride_cm, int32_t stride_cn, int32_t stride_am, int32_t stride_ak, int32_t stride_bk, int32_t stride_bn, int algo_id);
void load_matmul();
void unload_matmul();
    