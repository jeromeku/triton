
#include <cuda.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include "add_kernel.h"


static void write_buffer_to_csv(char *filename, int32_t *buffer, int size) {
   FILE *file = fopen(filename, "w");
   if (file == NULL) {
       printf("Could not open file %s\n", filename);
       return;
   }
   printf("Writing to %s\n", filename);
   for (int i = 0; i < size; i++) {
       fprintf(file, "%d", buffer[i]);
       if (i < size - 1) {
           fprintf(file, ",");
       }
   }
   fclose(file);
}

static void read_csv_to_buffer(char *filename, int32_t *buffer, int size) {
   FILE *file = fopen(filename, "r");
   if (file == NULL) {
       printf("Could not open file %s\n", filename);
       return;
   }
   int index = 0;
   printf("Reading from %s\n", filename);
   while (fscanf(file, "%d,", &buffer[index]) != EOF && index < size) {
       index++;
   }
   fclose(file);
}

int main(int argc, char **argv) {
  int N = 1024;

  // initialize CUDA handles
  CUdevice dev;
  CUcontext ctx;
  CUstream stream;
  CUdeviceptr x, y, out;
  CUresult err = 0;
  cuInit(0);
  cuDeviceGet(&dev, 0);
  cuCtxCreate(&ctx, 0, dev);
  cuMemAlloc(&x, N * 4);
  cuMemAlloc(&y, N * 4);
  cuMemAlloc(&out, N * 4);
  cuStreamCreate(&stream, 0);
  load_add_kernel();

  // initialize input data
  int32_t hx[N];
  int32_t hy[N];
  memset(hx, 0, N * 4);
  memset(hy, 0, N * 4);
  read_csv_to_buffer(argv[1], hx, N);
  read_csv_to_buffer(argv[2], hy, N);
  cuMemcpyHtoD(x, hx, N * 4);
  cuMemcpyHtoD(y, hy, N * 4);

  // launch kernel
  cuStreamSynchronize(stream);
  CUresult ret;
  int algo_id = 0;
  if (algo_id == 0) {
    ret = add_kernel_default(stream, x, y, out, N);
  } else {
    ret = add_kernel(stream, x, y, out, N, 0);
  }
  if (ret != 0) fprintf(stderr, "kernel launch failed\n");
  assert(ret == 0);

  cuStreamSynchronize(stream);

  // read data
  int32_t hout[N];
  memset(hout, 0, N * 4);
  cuMemcpyDtoH(hout, out, N * 4);    
  write_buffer_to_csv(argv[3], hout, N);

  // free cuda handles
  unload_add_kernel();
  cuMemFree(x);
  cuMemFree(y);
  cuMemFree(out);
  cuCtxDestroy(ctx);
}
