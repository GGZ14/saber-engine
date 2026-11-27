// Inkling 2 (inkling2_loop) prefill all-reduce wire, INK55 (docs/INKLING2.md section 9 INK55).
// Saber's own.
//
// The prefill reduce (`ink_p2p_xchg_v4`) PULLS the peer's f32 partial over the x8 link: 9.4 MB a
// 1,024-row reduce, 96 reduces a chunk, ~44 ms a 1,024-row chunk (CONS40's census). This object is
// the same reduce as a STAGED PUSH with a choice of wire form:
//
//   MODE 0  f32 raw                       byte-equal to v4: out = own + peer (one f32 add)
//   MODE 1  f32, exponent-group coded     byte-equal to v4 (the code is lossless)
//   MODE 2  bf16 raw                      out = f32(bf16(own)) + f32(bf16(peer))   (numerics)
//   MODE 3  bf16, exponent-group coded    MODE 2's bytes (the code is lossless over bf16)
//
// Both ranks compute the same expression with the operands swapped; an IEEE add is commutative,
// so both hold the same bytes (in MODE 2 / 3 each rank rounds ITS OWN partial too, by the same
// function, before the add).
//
// The wire. A BLOCK is 2,048 consecutive values of the [rows, cols] partial (one value octet a
// thread, 256 threads); a GROUP is 64 values (8 octets, 8 consecutive lanes). The coded forms
// send a group as one base exponent (the group's largest exponent field, in the block's header
// word) + 8 codes a lane:
//   f32:  sign(1) code(4) mantissa(23) = 28 bits, 8 a lane = 7 dwords (raw: 8)
//   bf16: sign(1) code(4) mantissa(7)  = 12 bits, 8 a lane = 3 dwords (raw: 4)
// code = base - e for an exponent field e >= 1 (0..14), and 15 for e == 0 (zero / subnormal: the
// mantissa is carried whole, so both are exact). A group whose nonzero exponent fields span more
// than 14 goes RAW (its header bit 8). A block in the window is [32 header dwords][the groups'
// payloads, compacted in group order]; a block's window stride is the raw size, so the block's
// address never depends on the data, only the bytes written over the link do.
//
// The stages. The blocks are cut into NP contiguous parts. Each workgroup pushes its blocks of
// part j, releases them at system scope, counts in on part j's counter (link word CNT + j), and
// the last arriver raises the PEER's arrive word ARR + j to the call's count s; then the
// workgroup folds part j - 1 (spins on OUR arrive word ARR + j - 1, the system-scope acquire in
// every reading thread -- FN71b: the peer's writes into our VRAM sit behind our caches otherwise)
// while part j's writes are still crossing. The window has two slots chosen by the call's GLOBAL
// count s (FN114d): our write of slot s & 1 at call s lands after the peer arrived (every part) at
// call s - 1, i.e. after it finished reading that slot at call s - 2 (kernel order on its stream).
// Link words (u32, from `link`): SEQ 32 (the call count), CNT 33..40, ARR 41..48 (the peer writes
// ours). Grid <= 128 workgroups (co-resident: a workgroup spins only after its own stores of the
// part), grid-stride over the part's blocks. Needs cols % 64 == 0, ld_own == ld_out == cols (the
// contiguous partial), 16-byte aligned bases (the host checks; otherwise it launches v4).
#include <hip/hip_runtime.h>

#define XW_SEQ 32
#define XW_CNT 33
#define XW_ARR 41
#define XW_MAXP 8
#define XW_BLK 2048

template <int MODE>
struct XwShape;
template <>
struct XwShape<0> {
  static constexpr int STRIDE = 2048;  // dwords a block in the window
  static constexpr int HDR = 0;
};
template <>
struct XwShape<1> {
  static constexpr int STRIDE = 32 + 2048;
  static constexpr int HDR = 32;
};
template <>
struct XwShape<2> {
  static constexpr int STRIDE = 1024;
  static constexpr int HDR = 0;
};
template <>
struct XwShape<3> {
  static constexpr int STRIDE = 32 + 1024;
  static constexpr int HDR = 32;
};

__device__ __forceinline__ unsigned xw_bf16(unsigned u) {
  // round to nearest even; a NaN stays a (quiet) NaN
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (u >> 16) | 0x40u;
  return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
}

__device__ __forceinline__ unsigned xw_grp_max(unsigned v) {
  v = max(v, (unsigned)__shfl_xor((int)v, 1, 8));
  v = max(v, (unsigned)__shfl_xor((int)v, 2, 8));
  v = max(v, (unsigned)__shfl_xor((int)v, 4, 8));
  return v;
}
__device__ __forceinline__ unsigned xw_grp_min(unsigned v) {
  v = min(v, (unsigned)__shfl_xor((int)v, 1, 8));
  v = min(v, (unsigned)__shfl_xor((int)v, 2, 8));
  v = min(v, (unsigned)__shfl_xor((int)v, 4, 8));
  return v;
}

// The wire values of one octet: f32 bit patterns (MODE 0 / 1) or bf16 patterns in the low 16 bits
// (MODE 2 / 3) of the partial at values [i, i + 8).
template <int MODE>
__device__ __forceinline__ void xw_load8(const float* __restrict__ p, long i, unsigned u[8]) {
  const uint4 a = *(const uint4*)&p[i];
  const uint4 b = *(const uint4*)&p[i + 4];
  u[0] = a.x; u[1] = a.y; u[2] = a.z; u[3] = a.w;
  u[4] = b.x; u[5] = b.y; u[6] = b.z; u[7] = b.w;
  if (MODE >= 2) {
#pragma unroll
    for (int k = 0; k < 8; ++k) u[k] = xw_bf16(u[k]);
  }
}

// Encode + push one block (b) of n values into the peer's window slot `dst` (dword pointer).
template <int MODE>
__device__ __forceinline__ void xw_push_block(const float* __restrict__ own, long n, long b,
                                              unsigned* dst, unsigned* s_blk, unsigned* s_mask) {
  constexpr int STRIDE = XwShape<MODE>::STRIDE;
  const int tid = threadIdx.x;
  const long base = b * XW_BLK;
  const long i = base + (long)tid * 8;
  const int ng = (int)min((long)32, (n - base) / 64);  // groups present (n % 64 == 0)
  const int g = tid >> 3, l8 = tid & 7;
  unsigned* out = dst + b * STRIDE;
  if (MODE == 0 || MODE == 2) {
    // raw: 16-byte stores straight to the peer, each store instruction one contiguous run of the
    // wave (f32: quad q = tid, tid + 256; a lane-strided pair of quads measured 1.7x slower)
    if (MODE == 0) {
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const int q = tid + 256 * k;
        if (base + (long)q * 4 < n) *(uint4*)&out[q * 4] = *(const uint4*)&own[base + (long)q * 4];
      }
      return;
    }
    if (g < ng) {
      unsigned u[8];
      xw_load8<MODE>(own, i, u);
      {
        *(uint4*)&out[tid * 4] = make_uint4(u[0] | (u[1] << 16), u[2] | (u[3] << 16),
                                            u[4] | (u[5] << 16), u[6] | (u[7] << 16));
      }
    }
    return;
  }
  // coded: exponent fields, the group's base and span
  constexpr int EB = (MODE == 1) ? 23 : 7;     // mantissa bits
  constexpr int WB = EB + 5;                   // code width: 28 / 12
  constexpr int PD = (MODE == 1) ? 7 : 3;      // payload dwords a lane, coded
  constexpr int RD = (MODE == 1) ? 8 : 4;      // raw
  unsigned u[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  unsigned emax = 0, emin = 255;
  const bool live = g < ng;
  if (live) {
    xw_load8<MODE>(own, i, u);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const unsigned e = (u[k] >> EB) & 0xFFu;
      emax = max(emax, e);
      if (e != 0) emin = min(emin, e);
    }
  }
  if (tid == 0) *s_mask = 0;
  emax = xw_grp_max(emax);
  emin = xw_grp_min(emin);
  const unsigned raw = (emin != 255u && emax - emin > 14u) ? 1u : 0u;
  __syncthreads();
  if (live && l8 == 0) {
    s_blk[g] = emax | (raw << 8);
    atomicOr(s_mask, raw << g);
  }
  __syncthreads();
  const unsigned mask = *s_mask;
  if (live) {
    const int off = 32 + PD * 8 * g + (RD - PD) * 8 * __popc(mask & ((1u << g) - 1u));
    if (raw) {
      unsigned* w = s_blk + off + l8 * RD;
      if (MODE == 1) {
#pragma unroll
        for (int k = 0; k < 8; ++k) w[k] = u[k];
      } else {
#pragma unroll
        for (int k = 0; k < 4; ++k) w[k] = u[2 * k] | (u[2 * k + 1] << 16);
      }
    } else {
      unsigned* w = s_blk + off + l8 * PD;
      unsigned long long acc = 0;
      int nb = 0, kd = 0;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const unsigned e = (u[k] >> EB) & 0xFFu;
        const unsigned code = e == 0 ? 15u : emax - e;
        const unsigned sgn = (u[k] >> (EB + 8)) & 1u;
        const unsigned c = (sgn << (WB - 1)) | (code << EB) | (u[k] & ((1u << EB) - 1u));
        acc |= (unsigned long long)c << nb;
        nb += WB;
        if (nb >= 32) {
          w[kd++] = (unsigned)acc;
          acc >>= 32;
          nb -= 32;
        }
      }
    }
  }
  __syncthreads();
  const int len = 32 + PD * 8 * ng + (RD - PD) * 8 * __popc(mask);  // dwords, % 4 == 0
  for (int q = tid; q < len / 4; q += 256) *(uint4*)&out[q * 4] = *(const uint4*)&s_blk[q * 4];
  __syncthreads();  // s_blk / s_mask reused by the next block
}

// Decode block b from our window `src` and fold: out = own' + peer'.
template <int MODE>
__device__ __forceinline__ void xw_fold_block(const float* __restrict__ own, long n, long b,
                                              const unsigned* src, float* __restrict__ outp,
                                              unsigned* s_hdr) {
  constexpr int STRIDE = XwShape<MODE>::STRIDE;
  const int tid = threadIdx.x;
  const long base = b * XW_BLK;
  const long i = base + (long)tid * 8;
  const int ng = (int)min((long)32, (n - base) / 64);
  const int g = tid >> 3, l8 = tid & 7;
  const bool live = g < ng;
  const unsigned* in = src + b * STRIDE;
  unsigned p[8];
  if (MODE == 0) {
    if (live) {
      const uint4 a = *(const uint4*)&in[tid * 8];
      const uint4 c = *(const uint4*)&in[tid * 8 + 4];
      p[0] = a.x; p[1] = a.y; p[2] = a.z; p[3] = a.w;
      p[4] = c.x; p[5] = c.y; p[6] = c.z; p[7] = c.w;
    }
  } else if (MODE == 2) {
    if (live) {
      const uint4 a = *(const uint4*)&in[tid * 4];
      const unsigned w[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        p[2 * k] = w[k] & 0xFFFFu;
        p[2 * k + 1] = w[k] >> 16;
      }
    }
  } else {
    constexpr int EB = (MODE == 1) ? 23 : 7;
    constexpr int WB = EB + 5;
    constexpr int PD = (MODE == 1) ? 7 : 3;
    constexpr int RD = (MODE == 1) ? 8 : 4;
    // wave 0's lane g reads group g's header; the raw mask by ballot (wave32: lane = group)
    if (__builtin_amdgcn_readfirstlane(tid >> 5) == 0) {
      const unsigned h = tid < ng ? in[tid] : 0u;
      s_hdr[tid] = h;
      const unsigned m = (unsigned)__ballot((h >> 8) & 1u);
      if (tid == 0) s_hdr[32] = m;
    }
    __syncthreads();
    if (live) {
      const unsigned h = s_hdr[g], mask = s_hdr[32];
      const unsigned emax = h & 0xFFu;
      const int off = 32 + PD * 8 * g + (RD - PD) * 8 * __popc(mask & ((1u << g) - 1u));
      if ((h >> 8) & 1u) {
        const unsigned* w = in + off + l8 * RD;
        if (MODE == 1) {
#pragma unroll
          for (int k = 0; k < 8; ++k) p[k] = w[k];
        } else {
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const unsigned x = w[k];
            p[2 * k] = x & 0xFFFFu;
            p[2 * k + 1] = x >> 16;
          }
        }
      } else {
        const unsigned* w = in + off + l8 * PD;
        unsigned long long acc = 0;
        int nb = 0, kd = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          if (nb < WB) {
            acc |= (unsigned long long)w[kd++] << nb;
            nb += 32;
          }
          const unsigned c = (unsigned)acc & ((1u << WB) - 1u);
          acc >>= WB;
          nb -= WB;
          const unsigned code = (c >> EB) & 15u;
          const unsigned e = code == 15u ? 0u : emax - code;
          p[k] = ((c >> (WB - 1)) << (EB + 8)) | (e << EB) | (c & ((1u << EB) - 1u));
        }
      }
    }
  }
  if (live) {
    unsigned o[8];
    xw_load8<MODE>(own, i, o);
    float v[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const unsigned a = MODE >= 2 ? (o[k] << 16) : o[k];
      const unsigned c = MODE >= 2 ? (p[k] << 16) : p[k];
      v[k] = __uint_as_float(a) + __uint_as_float(c);
    }
    *(float4*)&outp[i] = make_float4(v[0], v[1], v[2], v[3]);
    *(float4*)&outp[i + 4] = make_float4(v[4], v[5], v[6], v[7]);
  }
  if (MODE == 1 || MODE == 3) __syncthreads();  // s_hdr reused by the next block
}

template <int MODE>
__device__ __forceinline__ void xw_body(const float* __restrict__ own, unsigned* mywin,
                                        unsigned* peerwin, long slot_dw, float* __restrict__ out,
                                        int rows, int cols, int np, unsigned* link,
                                        unsigned* peer_link) {
  __shared__ __attribute__((aligned(16))) unsigned s_blk[XwShape<MODE>::HDR ? 32 + 2048 : 4];
  __shared__ unsigned s_mask;
  __shared__ unsigned s_last;
  const int tid = threadIdx.x;
  const unsigned s = __hip_atomic_load(&link[XW_SEQ], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) + 1u;
  const long slot = (long)(s & 1u) * slot_dw;
  const long n = (long)rows * cols;
  const long nblk = (n + XW_BLK - 1) / XW_BLK;
  unsigned* dst = peerwin + slot;
  const unsigned* src = mywin + slot;
  const long G = gridDim.x;
  if (tid == 0) s_last = 0;
  for (int j = 0; j <= np; ++j) {
    if (j < np) {
      const long b0 = nblk * j / np, b1 = nblk * (j + 1) / np;
      for (long b = b0 + blockIdx.x; b < b1; b += G)
        xw_push_block<MODE>(own, n, b, dst, s_blk, &s_mask);
      __threadfence_system();
      __syncthreads();
      if (__builtin_amdgcn_readfirstlane(tid >> 5) == 0) {
        unsigned last = 0;
        if (tid == 0) {
          const unsigned d =
              __hip_atomic_fetch_add(&link[XW_CNT + j], 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
          last = (d == (unsigned)G - 1u) ? 1u : 0u;
        }
        last = __builtin_amdgcn_readfirstlane(last);
        if (last && tid == 0) {
          __hip_atomic_store(&link[XW_CNT + j], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
          __atomic_thread_fence(__ATOMIC_RELEASE);
          __hip_atomic_store(&peer_link[XW_ARR + j], s, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
          if (j == np - 1) s_last = 1;
        }
      }
    }
    if (j >= 1) {
      const int jf = j - 1;
      if (__builtin_amdgcn_readfirstlane(tid >> 5) == 0) {
        while ((int)(__hip_atomic_load(&link[XW_ARR + jf], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) - s) < 0)
          __builtin_amdgcn_s_sleep(2);
      }
      __syncthreads();
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
      const long b0 = nblk * jf / np, b1 = nblk * (jf + 1) / np;
      for (long b = b0 + blockIdx.x; b < b1; b += G) xw_fold_block<MODE>(own, n, b, src, out, s_blk);
    }
  }
  // the next call's s: the last part's last arriver, once every workgroup counted in on every part
  // (each read link[SEQ] before its first count)
  __syncthreads();
  if (s_last && tid == 0) __hip_atomic_store(&link[XW_SEQ], s, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

#define XW_KERNEL(NAME, MODE)                                                                    \
  extern "C" __global__ __launch_bounds__(256) void NAME(                                        \
      const float* __restrict__ own, unsigned* mywin, unsigned* peerwin, long slot_dw,           \
      float* __restrict__ out, int rows, int cols, int np, unsigned* link, unsigned* peer_link) {  \
    xw_body<MODE>(own, mywin, peerwin, slot_dw, out, rows, cols, np, link, peer_link);           \
  }

XW_KERNEL(ink_xw55_f32, 0)
XW_KERNEL(ink_xw55_c32, 1)
XW_KERNEL(ink_xw55_b16, 2)
XW_KERNEL(ink_xw55_c16, 3)
