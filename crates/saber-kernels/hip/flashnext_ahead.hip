// Flash-Next (FN119 round 2, `--fn-device-step ahead`): THE ISSUE-AHEAD's device half -- the
// sampler + walk twin of `flashnext_dstep` (its source included verbatim: the same expf, noise,
// merge, masks and sequential sums) run on BOTH ranks, each reading the peer's seam half from its
// own P2P window; then the drafter round's acc-dependent inputs written on the card from the
// result block, so the round issues behind the verify before the host waits. Its own code object
// (`flashnext_dstep` keeps its bytes).
//
//   fn119a_push   grid (1), block 256: this rank's seam rows (the device pick's output) -> the PEER's
//                 window, slot = the call's global parity (`ctr` advanced once a call by thread 0;
//                 both ranks call in lockstep), every word stored INVERTED with relaxed system-scope
//                 stores (no flag: the data are the flag -- FN114b's law), then a system-scope
//                 write-back. A slot is rewritten two calls later, after the peer's walk of this call
//                 consumed (zeroed) it: the peer's next push follows its walk in stream order.
//   fn119a_walk   fn119_draw_walk with the peer half polled from this rank's window (a word not
//                 landed after FN119A_SPIN_MAX polls raises TIMEOUT in the result block: fail loud)
//                 and the row results (token, flags, argmax, 0).
//   fn119a_emb    grid (rows), block 256: the drafter's pass-1 token of verify row r (t <= acc: the
//                 walk's token t; t > acc: the row's argmax -- `Cmd::Commit`'s law) -> the payload
//                 row: the embedding row from this rank's half (UVA) or -0.0 (the other rank's id);
//                 the certified EXACT reduce sums the pair (x + -0.0 == x: FN22's law).
//   fn119a_gather grid (nseq), block 256: the drafter's stream row s * w + acc_s -> prev row s (the
//                 host's `mtp_gather` for pass 1).
//   fn119a_sel    grid (1), block 256: the chain inputs' variant of the walk's acc (sequence 0) ->
//                 the chain's input region (pinned variants read over UVA).
//   fn119a_backup grid (rows, 8), block 256: a row the walk refused (any flag) -> the pinned
//                 staging's raw row (UVA): the host's dense read after the round reused the arena.
#include "flashnext_dstep.hip"

#define FN119A_SPIN_MAX 200000000ULL
#define F119A_TIMEOUT 256u
#define FN119A_H 2560

extern "C" __global__ __launch_bounds__(256) void fn119a_push(const uint32_t* __restrict__ seam,
                                                             uint32_t* win_peer, uint32_t* ctr,
                                                             int rows, int rows_max) {
  __shared__ uint32_t s_c;
  if (threadIdx.x == 0) s_c = atomicAdd(ctr, 1u) + 1u;
  __syncthreads();
  const uint32_t c = s_c;
  uint32_t* dst = win_peer + (size_t)(c & 1u) * (size_t)rows_max * 2 * FN119_K;
  const int nw = rows * 2 * FN119_K;
  for (int i = threadIdx.x; i < nw; i += 256)
    __hip_atomic_store(&dst[i], ~seam[i], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  __threadfence_system();
}

extern "C" __global__ __launch_bounds__(32 * FN119_WMAX) void fn119a_walk(
    const uint32_t* __restrict__ seam_a, uint32_t* win, const uint32_t* __restrict__ ctr,
    const uint32_t* __restrict__ rp, const uint32_t* __restrict__ sq, uint32_t* __restrict__ out,
    uint32_t* __restrict__ rowout, int nseq, int rows_max) {
  __shared__ uint32_t s_id[FN119_WMAX][FN119_K];
  __shared__ float s_v[FN119_WMAX][FN119_K];
  __shared__ float s_e[FN119_WMAX][FN119_K];
  __shared__ float s_p[FN119_WMAX][FN119_K];
  __shared__ uint32_t s_ck[FN119_WMAX][2 * FN119_K];
  __shared__ uint32_t s_ci[FN119_WMAX][2 * FN119_K];
  __shared__ float s_sum[FN119_WMAX];
  __shared__ uint32_t s_tok[FN119_WMAX], s_fl[FN119_WMAX];
  __shared__ uint32_t s_to;
  const int b = blockIdx.x;
  if (threadIdx.x == 0) s_to = 0u;
  __syncthreads();
  if (b >= nseq) return;
  const int j = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const uint32_t* q = sq + (size_t)b * 8;
  const int w = (int)q[1];
  if (j < w) {
    const int row = (int)q[0] + j;
    const uint32_t* prm = rp + (size_t)row * 8;
    uint32_t* ids = s_id[j];
    float* v = s_v[j];
    float* e = s_e[j];
    float* p = s_p[j];
    uint32_t* ck = s_ck[j];
    uint32_t* ci = s_ci[j];
    // 1. the merge: 128 candidates, 4 a lane
    const size_t srow = (size_t)prm[5];
    const uint32_t* ha = seam_a + srow * 2 * FN119_K;
    // the peer's half: this call's slot of the window (the call's global parity), each word stored
    // INVERTED by the peer (an all-zero word = not landed: an id < 2^31 and a bf16's f32 bits never
    // invert to 0); polled at system scope, decoded, then zeroed for the slot's next use
    uint32_t* hb = win + (size_t)(ctr[0] & 1u) * (size_t)rows_max * 2 * FN119_K + srow * 2 * FN119_K;
    for (int c = lane; c < FN119_K; c += 32) {
      ci[c] = ha[2 * c];
      ck[c] = ha[2 * c + 1];
      uint32_t wi = 0, wk = 0;
      unsigned long long spins = 0;
      while ((wi == 0u || wk == 0u) && spins < FN119A_SPIN_MAX) {
        wi = __hip_atomic_load(&hb[2 * c], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        wk = __hip_atomic_load(&hb[2 * c + 1], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        ++spins;
      }
      if (wi == 0u || wk == 0u) atomicOr(&s_to, 1u);
      ci[FN119_K + c] = ~wi;
      ck[FN119_K + c] = ~wk;
      __hip_atomic_store(&hb[2 * c], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      __hip_atomic_store(&hb[2 * c + 1], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    WSYNC();
    for (int c = lane; c < 2 * FN119_K; c += 32) {
      const uint32_t kc = fn119_key(ck[c]), ic = ci[c];
      uint32_t rank = 0;
      for (int d = 0; d < 2 * FN119_K; ++d) {
        const uint32_t kd = fn119_key(ck[d]), id = ci[d];
        rank += (kd > kc || (kd == kc && id < ic)) ? 1u : 0u;
      }
      if (rank < FN119_K) {
        ids[rank] = ic;
        v[rank] = __uint_as_float(ck[c]);
      }
    }
    WSYNC();
    uint32_t fl = prm[4] & 2u ? F119_HOST : 0u;
    {
      const float a = v[lane], c = v[lane + 32];
      const bool bad = !(a - a == 0.0f) || !(c - c == 0.0f);  // NaN or inf
      fl |= fn119_wave_or(bad ? F119_NAN : 0u);
    }
    uint32_t tok = ids[0];
    if (!(prm[4] & 1u) && fl == 0u) {
      const float T = __uint_as_float(prm[0]);
      const float top_p = __uint_as_float(prm[1]);
      const float min_p = __uint_as_float(prm[2]);
      const uint32_t top_k = prm[3];
      const float floor_v = v[FN119_K - 1];
      WSYNC();
      // temperature
      v[lane] = __fdiv_rn(v[lane], T);
      v[lane + 32] = __fdiv_rn(v[lane + 32], T);
      WSYNC();
      const float ceiling = __fdiv_rn(floor_v, T);
      int tail = 0;  // 0 whole, 1 partial, 2 dead
      if (min_p > 0.0f) {
        float mx = -INFINITY;
        for (int i = 0; i < FN119_K; ++i) mx = fmaxf(mx, v[i]);
        tail = fn119_expf(ceiling - mx) < min_p ? 2 : (tail == 2 ? 2 : 1);
        fn119_softmax(v, e, p, &s_sum[j], lane);
        float pm = 0.0f;
        for (int i = 0; i < FN119_K; ++i) pm = fmaxf(pm, p[i]);
        const float thr = pm * min_p;
        if (p[lane] < thr) v[lane] = -INFINITY;
        if (p[lane + 32] < thr) v[lane + 32] = -INFINITY;
        WSYNC();
      }
      if (prm[4] & 4u) {  // top_k < vocab (the host's resolved knob)
        float thr;
        if (top_k == 0u || top_k >= (uint32_t)FN119_K) {
          thr = INFINITY;
          for (int i = 0; i < FN119_K; ++i) thr = fminf(thr, v[i]);
        } else {
          thr = v[top_k - 1];
        }
        tail = thr > ceiling ? 2 : (tail == 2 ? 2 : 1);
        if (top_k != 0u && top_k < (uint32_t)FN119_K) {
          WSYNC();
          if (v[lane] < thr) v[lane] = -INFINITY;
          if (v[lane + 32] < thr) v[lane + 32] = -INFINITY;
          WSYNC();
        }
      }
      if (top_p < 1.0f) {
        if (tail != 2) {
          fl |= F119_TAIL;
        } else {
          // ascending (value, index) ranks
          uint32_t* rk = ci;  // reuse: rank of entry i in the ascending order
          for (int c = lane; c < FN119_K; c += 32) {
            const float vc = v[c];
            uint32_t r = 0;
            for (int d = 0; d < FN119_K; ++d) r += (v[d] < vc || (v[d] == vc && d < c)) ? 1u : 0u;
            rk[c] = r;
          }
          WSYNC();
          uint32_t* ord = ck;  // ord[r] = i
          for (int c = lane; c < FN119_K; c += 32) ord[rk[c]] = (uint32_t)c;
          WSYNC();
          const float mx = v[ord[FN119_K - 1]];
          e[lane] = fn119_expf(v[lane] - mx);
          e[lane + 32] = fn119_expf(v[lane + 32] - mx);
          WSYNC();
          if (lane == 0) {
            float s = 0.0f;
            for (int r = 0; r < FN119_K; ++r) s += e[ord[r]];
            for (int i = 0; i < FN119_K; ++i) p[i] = __fdiv_rn(e[i], s);
            const float cut = 1.0f - top_p;
            float cum = 0.0f;  // tail.mass (dead) = 0
            const float scale = 1.0f - 0.0f;
            for (int r = 0; r < FN119_K; ++r) {
              const uint32_t i = ord[r];
              cum += p[i] * scale;
              if (r + 1 != FN119_K && cum <= cut) v[i] = -INFINITY;
            }
          }
          WSYNC();
        }
      }
      if (!(fl & F119_TAIL)) {
        const bool under = (v[lane] - v[lane] == 0.0f && v[lane] <= ceiling) ||
                           (v[lane + 32] - v[lane + 32] == 0.0f && v[lane + 32] <= ceiling);
        fl |= fn119_wave_or(under ? F119_CEIL : 0u);
      }
      if (fl == 0u) {
        fn119_softmax(v, e, p, &s_sum[j], lane);
        const uint64_t key = (uint64_t)prm[6] | ((uint64_t)prm[7] << 32);
        uint32_t g = 0;
        const float p0 = p[lane], p1 = p[lane + 32];
        const float e0 = fn119_noise(key, ids[lane], &g);
        uint32_t g0 = p0 > 0.0f ? g : 0u;
        g = 0;
        const float e1 = fn119_noise(key, ids[lane + 32], &g);
        uint32_t g1 = p1 > 0.0f ? g : 0u;
        fl |= fn119_wave_or((g0 | g1) ? F119_GUARDF : 0u);
        e[lane] = __fdiv_rn(p0, e0);
        e[lane + 32] = __fdiv_rn(p1, e1);
        WSYNC();
        if (lane == 0) {
          int best = 0;
          float bs = -INFINITY;
          for (int i = 0; i < FN119_K; ++i) {
            const float sc = e[i];
            if (sc > bs) {
              bs = sc;
              best = i;
            }
          }
          s_tok[j] = ids[best];
        }
        WSYNC();
        tok = s_tok[j];
      }
    }
    if (lane == 0) {
      s_tok[j] = tok;
      s_fl[j] = fl;
      rowout[(size_t)row * 4] = tok;
      rowout[(size_t)row * 4 + 1] = fl;
      rowout[(size_t)row * 4 + 2] = ids[0];
      rowout[(size_t)row * 4 + 3] = 0u;
    }
  }
  __syncthreads();
  // 2. the walk (drawn_rows + Greedy.accept)
  if (threadIdx.x == 0) {
    uint32_t* o = out + (size_t)b * 16;
    const int nd = (int)q[2];
    uint32_t sf = 0, n = 0;
    for (int r = 0; r < w; ++r) {
      sf |= s_fl[r];
      o[3 + r] = s_tok[r];
      ++n;
      if (r == nd || s_tok[r] != q[3 + r]) break;
    }
    for (int r = 0; r < FN119_WMAX; ++r) o[8 + r] = r < w ? s_fl[r] : 0u;
    for (int r = (int)n; r < FN119_WMAX; ++r) o[3 + r] = 0u;
    o[0] = sf | (s_to ? F119A_TIMEOUT : 0u);
    o[1] = n - 1u;
    o[2] = n;
    o[13] = o[14] = o[15] = 0u;
  }
}


extern "C" __global__ __launch_bounds__(256) void fn119a_emb(
    const uint32_t* __restrict__ out, const uint32_t* __restrict__ rowout,
    const uint16_t* __restrict__ embh, uint16_t* __restrict__ pay, int w, int nseq, int half,
    int rank) {
  const int r = blockIdx.x, s = r / w, t = r % w;
  uint32_t tok = 0u;  // a pad sequence's rows: token 0 (the host's law)
  if (s < nseq) {
    const uint32_t* o = out + (size_t)s * 16;
    const uint32_t acc = o[1];
    tok = (uint32_t)t <= acc ? o[3 + t] : rowout[(size_t)r * 4 + 2];
  }
  const uint32_t base = (uint32_t)rank * (uint32_t)half;
  const bool mine = tok >= base && tok < base + (uint32_t)half;
  uint4* dst = (uint4*)(pay + (size_t)r * FN119A_H);
  if (mine) {
    const uint4* src = (const uint4*)(embh + (size_t)(tok - base) * FN119A_H);
    for (int i = threadIdx.x; i < FN119A_H / 8; i += 256) dst[i] = src[i];
  } else {
    for (int i = threadIdx.x; i < FN119A_H / 8; i += 256)
      dst[i] = make_uint4(0x80008000u, 0x80008000u, 0x80008000u, 0x80008000u);
  }
}

extern "C" __global__ __launch_bounds__(256) void fn119a_gather(
    const uint32_t* __restrict__ out, const uint16_t* __restrict__ strm,
    uint16_t* __restrict__ prev, int w, int rowh) {
  const int s = blockIdx.x;
  const uint32_t acc = out[(size_t)s * 16 + 1];
  const uint4* src = (const uint4*)(strm + ((size_t)s * w + acc) * rowh);
  uint4* dst = (uint4*)(prev + (size_t)s * rowh);
  for (int i = threadIdx.x; i < rowh / 8; i += 256) dst[i] = src[i];
}

extern "C" __global__ __launch_bounds__(256) void fn119a_sel(const uint32_t* __restrict__ out,
                                                            const uint4* __restrict__ var,
                                                            uint4* __restrict__ dst, int stride16,
                                                            int n16) {
  const uint32_t acc = out[1];
  const uint4* src = var + (size_t)acc * stride16;
  for (int i = threadIdx.x; i < n16; i += 256) dst[i] = src[i];
}

extern "C" __global__ __launch_bounds__(256) void fn119a_backup(
    const uint32_t* __restrict__ rowout, const uint4* __restrict__ lg, uint4* stg, int row16) {
  const int r = blockIdx.x;
  if (rowout[(size_t)r * 4 + 1] == 0u) return;
  const int per = (row16 + (int)gridDim.y - 1) / (int)gridDim.y;
  const int a = (int)blockIdx.y * per, e = min(row16, a + per);
  for (int i = a + (int)threadIdx.x; i < e; i += 256)
    stg[(size_t)r * row16 + i] = lg[(size_t)r * row16 + i];
}
