// Flash-Next (FN68): the FP8 KV append (`--fn-kv-dtype fp8`). Its own code object, so
// `flashnext_glue` keeps its bytes.
//
//   fn_kv_append_ring_h256_fp8   fn_kv_append_ring_h256_bf16's job (the saber paged layout
//                                [block][kv head][slot][K 256 | V 256], the slot law, the
//                                indexer's raw-key ring) with K and V stored as e4m3 BYTES: the
//                                27B's `kv_append_h256_b16_fp8` law -- value / scale, the hardware
//                                convert (v_cvt_pk_fp8_f32, no clamp: a NaN stays a NaN), one
//                                PER-TENSOR scale for K and one for V (the host passes 1.0: the
//                                checkpoint carries no KV scales and the attention's descale
//                                pointers stay null = 1.0). The ring stays bf16 (the indexer's
//                                compressed keys are not the attention's pages).
//
// Strides are ELEMENTS of the page dtype (bytes here): the same numbers as the bf16 append's.
// Built with -ffp-contract=off (the `flashnext_glue` row of saber-kbuild's EXTRA_FLAGS).
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

typedef unsigned int u32;

// attn_glue.hip's e4m3_pair_raw, verbatim: divide, hardware convert, no clamp.
__device__ __forceinline__ u32 e4m3_pair_raw(float a, float b, float scale) {
  return __builtin_amdgcn_cvt_pk_fp8_f32(a / scale, b / scale, 0u, false) & 0xffffu;
}

// grid (m, nkv), 128 threads: thread t writes K pair t and V pair t (2 x 2 bytes).
extern "C" __global__ __launch_bounds__(128) void fn_kv_append_ring_h256_fp8(
    const __bf16 *__restrict__ key, long key_stride, const __bf16 *__restrict__ value,
    long value_stride, unsigned char *__restrict__ kv, const long *__restrict__ slot,
    long block_stride, long page_stride, long head_stride, const __bf16 *__restrict__ ikey,
    long ikey_stride, __bf16 *__restrict__ ring, long *__restrict__ ring_pos,
    const long *__restrict__ ring_slot, const long *__restrict__ pos, int m, float k_scale,
    float v_scale) {
  const int t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
  const long s = slot[t];
  if (s >= 0) {
    const long blk = s / 16, off = s % 16;
    const __bf16 *kp = key + (long)t * key_stride + (long)h * 256;
    const __bf16 *vp = value + (long)t * value_stride + (long)h * 256;
    unsigned char *base = kv + blk * block_stride + (long)h * head_stride + off * page_stride;
    const int d = tid * 2;
    const u32 kq = e4m3_pair_raw((float)kp[d], (float)kp[d + 1], k_scale);
    const u32 vq = e4m3_pair_raw((float)vp[d], (float)vp[d + 1], v_scale);
    *(unsigned short *)(base + d) = (unsigned short)kq;
    *(unsigned short *)(base + 256 + d) = (unsigned short)vq;
  }
  if (h != 0) return;
  const long rs = ring_slot[t];
  if (rs < 0) return;
  if (tid < 64) ((u32 *)(ring + rs * 128))[tid] = ((const u32 *)(ikey + (long)t * ikey_stride))[tid];
  if (tid < 3) ring_pos[rs * 3 + tid] = pos[(long)tid * m + t];
}
