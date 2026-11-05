// FN64c probe kernels: what a captured graph exec costs in VRAM on ROCm 10.1.
// k_nop: 8-byte kernarg, no scratch. k_big: 512-byte kernarg. k_scratch: 4 KiB private segment.
#include <hip/hip_runtime.h>
extern "C" __global__ void k_nop(int* p) {
  if (threadIdx.x == 0 && p) p[blockIdx.x] += 1;
}
struct Big { long v[64]; };
extern "C" __global__ void k_big(Big b) {
  if (threadIdx.x == 0 && b.v[0]) ((int*)b.v[0])[blockIdx.x] += (int)b.v[63];
}
extern "C" __global__ void k_scratch(int* p, int n) {
  volatile int buf[1024];
  for (int i = 0; i < 1024; i++) buf[(i * n) & 1023] = i + n;
  if (p) p[threadIdx.x] = buf[(threadIdx.x * n) & 1023];
}
