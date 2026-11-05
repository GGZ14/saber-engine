// The drafter's KV append, at head_dim 128 and in the reference server's FLASH layout.
//
// WHERE THIS BELONGS, and why it is here instead. The `attn_glue` plugin already has a `kv_append`
// gated bit-exact against this same `reshape_and_cache_flash` at head_dim 256 (ws-f-saber, round 7),
// and the round that asked for this one asked for an INSTANTIATION of it. That commit (368d3c8) is
// not in main -- checked, not assumed: of ws-f-the reference's rounds 6-11 only 22f0223 is an ancestor of
// main, so `attn_glue.hip` does not exist on this base. Rather than merge another branch's pending
// work into this one, the kernel lands in the drafter's own plugin; folding it into `attn_glue` as a
// second instantiation once that plugin is in main is a rename and a rules row.
//
// THE LAYOUT IS THE POINT. The drafter's cache is `[blocks, block_size, kv_heads, head_size]` with
// SEPARATE k and v tensors -- the reference server's flash layout -- while the reference's paged cache interleaves K and V
// per token in one buffer. That difference is why the drafter's attention cannot be an
// instantiation of `saber_attn`, and it is stated in the manifest signature so a plan that hands this
// kernel a saber-layout cache is refused rather than silently writing into the wrong halves.
//
//   slot = slot_mapping[i]           -- a BLOCK TABLE WALK, not the token index. Prod's slots are
//                                       non-contiguous and a kernel that assumes otherwise passes a
//                                       contiguous test and corrupts a serve.
//
// AND THE SOURCE ROWS HAVE STRIDES, which this kernel did not take until F29. The reference server's own
// `reshape_and_cache_flash` takes `key_stride` and `value_stride` because the caller hands it
// VIEWS: the drafter's k comes out of the rope as a contiguous `[M, 8, 128]` tensor and its **v is
// a view of the `[q | k | v]` projection** (`reinterpret_tensor(buf36, (s72, 8, 128), (6144, 128,
// 1), 5120)` in prod's own graph, and the capture's manifest records the same 6144). Assuming
// `i * nkv * head` for both read v row 0 correctly and then walked into the NEXT ROW'S Q. It cost
// 7,121 of 3,375,104 cache bytes -- every appended v byte -- and a staged replay could never see
// it, because it seeds the attention from prod's own cache.
//   slot < 0                         -- skipped, which is how the reference server marks a padded token
//   cache[slot / block_size][slot % block_size][h][d] = key[i][h][d]
// and for an fp8 cache, `scaled_convert` divides by the scale before the e4m3 round (1.0 when the
// serve is uncalibrated, which is prod's state).
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#define DK_NT 256

// Two source rows (key and value) are written per token, so one launch does both caches.
template <bool FP8>
__device__ __forceinline__ void dk_append(const __bf16 *__restrict__ key,
                                          const __bf16 *__restrict__ value,
                                          void *__restrict__ kc, void *__restrict__ vc,
                                          const long *__restrict__ slots,
                                          const float *__restrict__ scales, int rows, int nkv,
                                          int head, int block_size, int key_stride,
                                          int value_stride) {
  const int i = (int)blockIdx.x;
  if (i >= rows) return;
  const long slot = slots[i];
  if (slot < 0) return;                       // the reference server's padded-token marker
  const long blk = slot / block_size, off = slot % block_size;
  const long dst = ((blk * block_size + off) * nkv) * (long)head;
  const long ksrc = (long)i * key_stride;
  const long vsrc = (long)i * value_stride;
  const int n = nkv * head;
  const float ks = FP8 ? scales[0] : 1.f, vs = FP8 ? scales[1] : 1.f;
  for (int e = (int)threadIdx.x; e < n; e += DK_NT) {
    if (FP8) {
      // `scaled_convert`: divide by the scale, then one e4m3 round. Two at a time, because
      // `cvt_pk_fp8_f32` is the only e4m3 encoder this tree uses (mxfp4_quant.hip:109).
      const float a = (float)key[ksrc + e] / ks;
      const float b = (float)value[vsrc + e] / vs;
      ((unsigned char *)kc)[dst + e] =
          (unsigned char)(__builtin_amdgcn_cvt_pk_fp8_f32(a, a, 0u, false) & 0xffu);
      ((unsigned char *)vc)[dst + e] =
          (unsigned char)(__builtin_amdgcn_cvt_pk_fp8_f32(b, b, 0u, false) & 0xffu);
    } else {
      ((__bf16 *)kc)[dst + e] = key[ksrc + e];
      ((__bf16 *)vc)[dst + e] = value[vsrc + e];
    }
  }
}

extern "C" __global__ __launch_bounds__(DK_NT) void dflash_kv_append_flash_bf16(
    const __bf16 *__restrict__ key, const __bf16 *__restrict__ value, __bf16 *__restrict__ kc,
    __bf16 *__restrict__ vc, const long *__restrict__ slots, const float *__restrict__ scales,
    int rows, int nkv, int head, int block_size, int key_stride, int value_stride) {
  dk_append<false>(key, value, kc, vc, slots, scales, rows, nkv, head, block_size, key_stride,
                   value_stride);
}

extern "C" __global__ __launch_bounds__(DK_NT) void dflash_kv_append_flash_fp8(
    const __bf16 *__restrict__ key, const __bf16 *__restrict__ value,
    unsigned char *__restrict__ kc, unsigned char *__restrict__ vc,
    const long *__restrict__ slots, const float *__restrict__ scales, int rows, int nkv, int head,
    int block_size, int key_stride, int value_stride) {
  dk_append<true>(key, value, kc, vc, slots, scales, rows, nkv, head, block_size, key_stride,
                  value_stride);
}
