// Flash-Next (LR-N, `--spec-couple on`): COUPLED DRAFTING -- the MTP drafter's chain tail drafts a
// sampled row's token with the TARGET's own noise at the position the draft will be verified at.
// Its own code object: every shipped object keeps its bytes.
//
// THE LAW (Saber's own; the math, not anyone's code):
//   The target's sampled draw at output position p is the exponential race
//       t = argmax_i  p_i / E_i,   E_i = -ln u(key_p, i)   (the host sampler's SeededNoise;
//                                                         FN119's fn119_noise on the card)
//   over its processed distribution p (temperature, top_k, top_p). The drafter proposes
//       x = argmax_{i in C}  s_i / T - ln E_i              (the SAME E_i, the same key_p)
//   over its own candidates C (this rank's top-`top_k` of the int2 head's reranked candidates, the
//   pair's union by the larger perturbed key), s = the drafter's exact rescored logit, T = the
//   request's temperature. The walk is UNCHANGED (row j keeps the target's draw; it continues while
//   the draw IS the draft), so the committed tokens are the target's draws, byte for byte, with the
//   arm on or off: the output law, every seed's tokens and every byte gate stand. Only the DRAFTS
//   move, toward the target's draw: acceptance P(x == t) instead of p(argmax s).
//   A greedy row (invT == 0), a pad row and a row without coupled state keep FN22's integer argmax
//   (the 16-bit bf16 key in the low half of a 32-bit key): the shipped drafts, bit for bit.
//
// The chain input buffer `inb` (u32 words; the host's `chain_input_bytes`):
//   word 0             the pass counter (pass 1 = 0; fn_couple_select advances it)
//   words 20 + 4r ..   row r: key base lo, key base hi (mix(seed) ^ mix(choice * C2)), p1 (the
//                      output position pass 1 drafts), invT bits (0 = greedy: FN22's argmax)
//   word 52 + r        row r: top_k (0 = every candidate)
//   pass c drafts position p1 + c: key = base ^ mix((p1 + c) * C1) == SeededNoise's key.
//
//   fn_couple_acp     grid (rows), block 256 == FN86d's fused argmax + pack (wave 0 picks, every
//                     thread packs) with the coupled key: lanes (k31..24, k23..16, k15..8, k7..0,
//                     gid b0, b1, b2, 0) per rank -- a 32-bit key.
//   fn_couple_select  grid (1), block 256 == fn_draft_select over the 32-bit keys (the larger key;
//                     a tie -> rank 0's candidate).
//
// Built with -ffp-contract=off (the table's row; drafting is free, the flag keeps the line uniform).
#include <hip/hip_runtime.h>
#include <stdint.h>

#define FNC_SPLITS 32
#define FNC_H 2560
#define FNC_LANES 16
#define FNC_IDS_STRIDE 8
#define FNC_KMAX 128
#define FNC_W_ROWS 20
#define FNC_W_TOPK 52

__device__ __forceinline__ uint32_t fnc_key16(uint32_t b) {
  return (b & 0x8000u) ? (~b & 0xffffu) : (b | 0x8000u);
}

// the f32 total-order key (larger = larger value)
__device__ __forceinline__ uint32_t fnc_key32(float f) {
  const uint32_t b = __float_as_uint(f);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__device__ __forceinline__ unsigned long long fnc_max(unsigned long long a, unsigned long long b) {
  return a > b ? a : b;
}

__device__ __forceinline__ unsigned long long fnc_wave_max(unsigned long long v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const uint32_t lo = __shfl_xor((uint32_t)v, o, 32);
    const uint32_t hi = __shfl_xor((uint32_t)(v >> 32), o, 32);
    v = fnc_max(v, ((unsigned long long)hi << 32) | lo);
  }
  return v;
}

__device__ __forceinline__ uint16_t fnc_enc(uint32_t v) {
  return (uint16_t)(__builtin_bit_cast(uint32_t, (float)v) >> 16);
}

__device__ __forceinline__ uint32_t fnc_dec(uint16_t b) {
  return (uint32_t)__builtin_bit_cast(float, (uint32_t)b << 16);
}

// splitmix64 (the sampler's mixing function)
__device__ __forceinline__ uint64_t fnc_mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  uint64_t z = x;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// E(id) = -(ln u) as f32, u = ((mix(key ^ mix(id)) >> 11) + 1) * 2^-53 (the target's noise)
__device__ __forceinline__ float fnc_noise(uint64_t key, uint32_t id) {
  const uint64_t bits = fnc_mix(key ^ fnc_mix((uint64_t)id));
  const double u = (double)((bits >> 11) + 1ull) * __longlong_as_double(0x3ca0000000000000ll);
  return (float)(-log(u));
}

extern "C" __global__ __launch_bounds__(256) void fn_couple_acp(
    const uint16_t *__restrict__ sc, const int *__restrict__ cand, const int *__restrict__ sel,
    unsigned long long *__restrict__ part, int kk, int ldc, int cstride,
    const uint16_t *__restrict__ embh, uint16_t *__restrict__ pay, int half, int rank,
    const uint32_t *__restrict__ inb) {
  __shared__ uint32_t s_k[FNC_KMAX];
  __shared__ uint32_t s_c[FNC_KMAX];
  __shared__ uint32_t win[2];
  const int r = blockIdx.x, tid = threadIdx.x;
  const uint32_t *cp = inb + FNC_W_ROWS + 4 * r;
  const uint32_t invt_bits = cp[3];
  const bool live = sel[r] >= 0;
  const bool coupled = live && invt_bits != 0u && kk <= FNC_KMAX;  // workgroup-uniform
  if (tid < 32) {
    unsigned long long best = 0;
    if (coupled) {
      for (int j = tid; j < kk; j += 32) {
        s_k[j] = fnc_key16(sc[(long)r * ldc + r * kk + j]);
        s_c[j] = (uint32_t)cand[r * cstride + j];
      }
      __builtin_amdgcn_wave_barrier();
      __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
      const uint32_t tk = inb[FNC_W_TOPK + r];
      const uint64_t base = (uint64_t)cp[0] | ((uint64_t)cp[1] << 32);
      const uint64_t pos = (uint64_t)cp[2] + (uint64_t)inb[0];
      const uint64_t key = base ^ fnc_mix(pos * 0x00FF51AFD7ED558Dull);
      const float invt = __uint_as_float(invt_bits);
      for (int j = tid; j < kk; j += 32) {
        const uint32_t kj = s_k[j], cj = s_c[j];
        uint32_t rnk = 0;
        for (int d = 0; d < kk; ++d) {
          const uint32_t kd = s_k[d], cd = s_c[d];
          rnk += (kd > kj || (kd == kj && cd < cj)) ? 1u : 0u;
        }
        if (tk == 0u || rnk < tk) {
          const float v = __uint_as_float((uint32_t)sc[(long)r * ldc + r * kk + j] << 16);
          const uint32_t gid = (uint32_t)rank * (uint32_t)half + cj;
          const float e = fnc_noise(key, gid);
          const float z = v * invt - __logf(e);
          if (z == z) best = fnc_max(best, ((unsigned long long)fnc_key32(z) << 32) | ~cj);
        }
      }
    } else if (live) {
      // FN22's integer argmax (the 16-bit key, the lower index on a tie)
      for (int j = tid; j < kk; j += 32) {
        const uint32_t b = sc[(long)r * ldc + r * kk + j];
        best = fnc_max(best, ((unsigned long long)fnc_key16(b) << 32) |
                                 ~(uint32_t)cand[r * cstride + j]);
      }
    }
    best = fnc_wave_max(best);
    part[r * FNC_SPLITS + tid] = tid == 0 ? best : 0ull;
    if (tid == 0) {
      uint32_t key = (uint32_t)(best >> 32);
      uint32_t lidx = ~(uint32_t)best;
      if (!live) {  // a pad row: token 0 (rank 0's row 0) wins the pair
        key = rank == 0 ? 0xffffffffu : 0u;
        lidx = 0;
      }
      win[0] = key;
      win[1] = lidx;
    }
  }
  __syncthreads();
  const uint32_t key = win[0], lidx = win[1];
  uint16_t *row = pay + (long)r * (2 * FNC_H + FNC_LANES);
  uint4 *mine = (uint4 *)(row + rank * FNC_H);
  uint4 *other = (uint4 *)(row + (1 - rank) * FNC_H);
  const uint4 *src = (const uint4 *)(embh + (long)lidx * FNC_H);
  const uint4 nz = make_uint4(0x80008000u, 0x80008000u, 0x80008000u, 0x80008000u);
  for (int i = tid; i < FNC_H / 8; i += 256) {
    mine[i] = src[i];
    other[i] = nz;
  }
  if (tid < FNC_LANES) {
    const int owner = tid >> 3, j = tid & 7;
    uint16_t b = 0x8000;
    if (owner == rank) {
      const uint32_t gid = (uint32_t)rank * (uint32_t)half + lidx;
      const uint32_t byte = j <= 3 ? (key >> (8 * (3 - j))) & 0xffu
                            : j <= 6 ? (gid >> (8 * (j - 4))) & 0xffu
                                     : 0u;
      b = fnc_enc(byte);
    }
    row[2 * FNC_H + tid] = b;
  }
}

extern "C" __global__ __launch_bounds__(256) void fn_couple_select(
    const uint16_t *__restrict__ red, uint16_t *__restrict__ emb, uint32_t *__restrict__ ids,
    uint32_t *__restrict__ ctr, int rows) {
  const int tid = threadIdx.x;
  const uint32_t c = *ctr;
  for (int r = 0; r < rows; ++r) {
    const uint16_t *row = red + (long)r * (2 * FNC_H + FNC_LANES);
    const uint16_t *ln = row + 2 * FNC_H;
    uint32_t k0 = 0, k1 = 0;
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      k0 = (k0 << 8) | fnc_dec(ln[b]);
      k1 = (k1 << 8) | fnc_dec(ln[8 + b]);
    }
    const int w = k1 > k0;
    const uint16_t *l = ln + 8 * w;
    const uint32_t gid = fnc_dec(l[4]) | (fnc_dec(l[5]) << 8) | (fnc_dec(l[6]) << 16);
    const uint4 *src = (const uint4 *)(row + w * FNC_H);
    uint4 *dst = (uint4 *)(emb + (long)r * FNC_H);
    for (int i = tid; i < FNC_H / 8; i += 256) dst[i] = src[i];
    if (tid == 0) ids[c * FNC_IDS_STRIDE + r] = gid;
  }
  __syncthreads();
  if (tid == 0) *ctr = c + 1;
}
