// THE ALL-REDUCE RACE CHECKS (workstream T4). Hand-written, device-only. Used by the check code
// objects (`saber_ar_exact_chk`, `saber_ar_wht_chk`, `saber_ar_wht_wpar_chk` and the `_chk` symbols of
// `flashnext_ar` / `flashnext_ar_gp`, through the CR2-4c bodies `sar_*.h` at their checked level; and
// `mxfp4_gemm_decode_arnq_chk.hip`), which are the served all-reduce kernels with these helpers
// called around them and NOTHING else changed: the pushes, the handshake, the reduce and the
// epilogue are the served code, so a checked run computes the served bytes (the host gate
// `ar_gate --race-check` holds it).
//
// What is checked, per call and per block (or per GEMM tile piece), with no retry that feeds the
// arithmetic -- an event is COUNTED and the first one is RECORDED, and the kernel goes on with the
// bytes it read, so a checked boot still shows the event in its digests:
//
//  * THE PAYLOAD. The sender XORs a position-keyed hash of every 16-byte word it pushes, mixed with
//    the call's sequence number, into one check word per (block, piece) at the TAIL of the peer's
//    slot (`CHK_RESERVE_BYTES` below the slot's end; the host refuses a message that would reach
//    it). The receiver hashes what it READ after the flag and compares. A mismatch is a read of
//    data the peer had not landed (or of the other call's slot): the one thing the handshake exists
//    to prevent. The first mismatch's (seq, block, piece) are kept; a mismatch then RE-READS the
//    piece at system scope until it matches or `CHK_PROBE_MAX` polls pass, which separates "the
//    data landed late" (converges) from "this is another call's data" (never converges). The
//    re-read is a diagnostic only: the reduce already used the first read.
//  * THE FLAG. After the spin the receiver re-reads its flag: `== s` is the lockstep the protocol
//    assumes, `s + 1` is the peer already in the next call (benign), anything else is DRIFT; a
//    spin that ran out (`SABER_AR_SPIN_MAX`) is counted apart.
//  * THE SPLIT-K COUNTER (row 3's GEMM): an election counter found un-reset at arrival.
//
// Everything lands in `seq_ctrs + CHK_DIAG` (the `ar_seq` region is 4096 B under the check --
// `saber::AR_CHECK_SEQ_BYTES`, the served 2048 B otherwise; the served kernels use its first 512
// words), which is LOCAL memory the peer never writes. The engine reads it after every step under
// `--engine-race-check` and logs each new event with its step.
#pragma once

#define CHK_DIAG 512u
#define CHK_PIECES 128u
#define CHK_RESERVE_BYTES (512L * 128L * 4L)
// The re-read's bound: CHK_PROBE_MAX polls, each after a ~0.2 us s_sleep, the whole piece re-read
// by the WHOLE block each poll -- a never-converging mismatch (the control: a peer that writes
// no check word) costs ~CHK_PROBE_MAX x (sleep + one read of the piece), well under a millisecond.
#define CHK_PROBE_MAX 256u

// The diagnostic words, as the host decodes them (`saber_exec::race::ArDiag`).
#define CHK_D_BLOCKS 0      // blocks (or row owners) checked
#define CHK_D_MISMATCH 1    // first-read hash != the sender's check word
#define CHK_D_CONVERGED 2   // ... and a system-scope re-read matched within CHK_PROBE_MAX polls
#define CHK_D_NEVER 3       // ... and it never did
#define CHK_D_AHEAD 4       // flag == s + 1 after the spin (the peer is in the next call: benign)
#define CHK_D_DRIFT 5       // flag neither s nor s + 1 after the spin
#define CHK_D_TIMEOUT 6     // the spin ran out
#define CHK_D_CNT 7         // row 3: a split-K election counter found un-reset
#define CHK_D_FIRST_S 8     // the first mismatch's sequence number
#define CHK_D_FIRST_B 9     // ... its block (row)
#define CHK_D_FIRST_P 10    // ... its piece (row 3: the GEMM n-tile)
#define CHK_D_POLLS 11      // the most re-read polls a converged mismatch took
#define CHK_D_LAST_S 12     // block 0's (row 0's) last sequence number: the call counter

__device__ __forceinline__ unsigned chk_fmix(unsigned h) {
  h ^= h >> 16;
  h *= 0x85ebca6bu;
  h ^= h >> 13;
  h *= 0xc2b2ae35u;
  h ^= h >> 16;
  return h;
}

// A 16-byte word at position `k`: every lane and the position enter, so a word read at the wrong
// place, a torn word and a stale word each change it.
__device__ __forceinline__ unsigned chk_word(int4 v, unsigned k) {
  const unsigned p = k * 0x9e3779b1u;
  return chk_fmix((unsigned)v.x + p) ^ chk_fmix((unsigned)v.y + p + 0x7f4a7c15u) ^
         chk_fmix((unsigned)v.z + p + 0x1b873593u) ^ chk_fmix((unsigned)v.w + p + 0xcc9e2d51u);
}

__device__ __forceinline__ unsigned chk_word4(uint4 v, unsigned k) {
  return chk_word(make_int4((int)v.x, (int)v.y, (int)v.z, (int)v.w), k);
}

// A 16-bit scale at group `g` (the rotated reduce's bf16 scales).
__device__ __forceinline__ unsigned chk_half(unsigned short v, unsigned g) {
  return chk_fmix(((unsigned)v << 16) ^ (g * 0x9e3779b1u) ^ 0x5a5au);
}

__device__ __forceinline__ unsigned chk_seq(unsigned s) {
  return chk_fmix(s * 0x9e3779b1u + 0x7f4a7c15u);
}

// The check words of one slot: its last CHK_RESERVE_BYTES.
__device__ __forceinline__ unsigned* chk_ck(const void* slot_base, long slot_bytes) {
  return (unsigned*)((char*)slot_base + slot_bytes - CHK_RESERVE_BYTES);
}

// A load the reduce uses, hashed on the way.
__device__ __forceinline__ int4 chk_ld_at(const int4* p, int k, unsigned at, unsigned& h) {
  const int4 v = p[k];
  h ^= chk_word(v, at);
  return v;
}

// The re-read: system scope, one dword at a time, around every cache.
__device__ __forceinline__ unsigned chk_ld32_sys(const unsigned* p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ int4 chk_ld_sys(const int4* p) {
  const unsigned* q = (const unsigned*)p;
  return make_int4((int)chk_ld32_sys(q), (int)chk_ld32_sys(q + 1), (int)chk_ld32_sys(q + 2),
                   (int)chk_ld32_sys(q + 3));
}

// XOR over the whole block; every thread must call it (it meets twice).
__device__ __forceinline__ unsigned chk_block_xor(unsigned h) {
  __shared__ unsigned chk_s_x[32];
  for (int o = 16; o; o >>= 1) h ^= __shfl_xor(h, o);
  const int w = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
  __syncthreads();
  if ((threadIdx.x & 31) == 0) chk_s_x[w] = h;
  __syncthreads();
  unsigned t = 0u;
  for (int i = 0; i < nw; ++i) t ^= chk_s_x[i];
  return t;
}

// XOR over each aligned group of 16 lanes (row 3's pieces: one row's 16 words of one n-tile).
__device__ __forceinline__ unsigned chk_xor16(unsigned h) {
  for (int o = 8; o; o >>= 1) h ^= __shfl_xor(h, o);
  return h;
}

// Sender: one check word per (block, piece), written into the PEER's slot tail before the drain.
__device__ __forceinline__ void chk_send(unsigned h, const void* peer_slot, long slot_bytes,
                                         unsigned b, unsigned piece, unsigned s) {
  const unsigned t = chk_block_xor(h);
  if (threadIdx.x == 0) chk_ck(peer_slot, slot_bytes)[b * CHK_PIECES + piece] = t ^ chk_seq(s);
}

// tid 0, right after the spin.
__device__ __forceinline__ void chk_after_spin(const unsigned* flag, unsigned s,
                                               unsigned long long z, unsigned* diag) {
  const unsigned v = __hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
  if (z > SABER_AR_SPIN_MAX) atomicAdd(&diag[CHK_D_TIMEOUT], 1u);
  if (v == s + 1u) atomicAdd(&diag[CHK_D_AHEAD], 1u);
  else if (v != s) atomicAdd(&diag[CHK_D_DRIFT], 1u);
}

__device__ __forceinline__ void chk_mismatch(unsigned* diag, unsigned s, unsigned b, unsigned p) {
  if (atomicAdd(&diag[CHK_D_MISMATCH], 1u) == 0u) {
    diag[CHK_D_FIRST_S] = s;
    diag[CHK_D_FIRST_B] = b;
    diag[CHK_D_FIRST_P] = p;
  }
}

__device__ __forceinline__ void chk_probe_done(unsigned* diag, unsigned polls, bool ok) {
  if (ok) {
    atomicAdd(&diag[CHK_D_CONVERGED], 1u);
    atomicMax(&diag[CHK_D_POLLS], polls);
  } else {
    atomicAdd(&diag[CHK_D_NEVER], 1u);
  }
}

// Receiver, a contiguous run of words [start, end) that is ONE piece (the exact kernels: a
// block's chunk; `_nq`: a row). Every thread calls it; on a mismatch the whole block re-reads the
// piece (each thread its own words, at system scope) until it matches or CHK_PROBE_MAX polls pass.
__device__ __forceinline__ void chk_recv(unsigned h, const int4* ms, int start, int end, unsigned b,
                                         unsigned piece, unsigned s, long slot_bytes,
                                         unsigned* diag, bool rel) {
  __shared__ int chk_bad;
  const unsigned t = chk_block_xor(h);
  const unsigned* ck = chk_ck(ms, slot_bytes) + b * CHK_PIECES + piece;
  if (threadIdx.x == 0) {
    if (b == 0u) diag[CHK_D_LAST_S] = s;
    atomicAdd(&diag[CHK_D_BLOCKS], 1u);
    chk_bad = (t ^ chk_seq(s)) != *ck;
    if (chk_bad) chk_mismatch(diag, s, b, piece);
  }
  __syncthreads();
  if (!chk_bad) return;  // uniform across the block
  unsigned polls = 0u;
  bool ok = false;
  while (!ok && polls < CHK_PROBE_MAX) {
    ++polls;
    __builtin_amdgcn_s_sleep(8);
    unsigned r = 0u;
    for (int k = start + (int)threadIdx.x; k < end; k += (int)blockDim.x)
      r ^= chk_word(chk_ld_sys(ms + k), rel ? (unsigned)(k - start) : (unsigned)k);
    const unsigned tt = chk_block_xor(r);
    if (threadIdx.x == 0) chk_bad = (tt ^ chk_seq(s)) != chk_ld32_sys(ck);
    __syncthreads();
    ok = !chk_bad;
    __syncthreads();
  }
  if (threadIdx.x == 0) chk_probe_done(diag, polls, ok);
}

#ifdef SABER_ARNQ_MAXG
// Row 3's receiver (`mxfp4_gemm_decode_arnq_chk`): the row arrived as one piece per GEMM n-tile
// (16 words each), so each 16-lane group of the owner block checks the piece its lanes read, at
// the word index WITHIN THE ROW (what the tile hashed). Every thread calls it.
__device__ __forceinline__ void chk_recv_pieces(const unsigned (&hp)[SABER_ARNQ_MAXG], const int4* ms,
                                                int start, int end, unsigned b, unsigned s,
                                                long slot_bytes, unsigned* diag) {
  const int tid = threadIdx.x;
  if (tid == 0) {
    if (b == 0u) diag[CHK_D_LAST_S] = s;
    atomicAdd(&diag[CHK_D_BLOCKS], 1u);
  }
#pragma unroll
  for (int g = 0; g < SABER_ARNQ_MAXG; ++g) {
    const unsigned h = chk_xor16(hp[g]);
    const int off = tid + g * SABER_ARNQ_T;
    if ((tid & 15) == 0 && start + off < end) {
      const unsigned piece = (unsigned)(off >> 4);
      const unsigned* ck = chk_ck(ms, slot_bytes) + b * CHK_PIECES + piece;
      if ((h ^ chk_seq(s)) != *ck) {
        chk_mismatch(diag, s, b, piece);
        unsigned polls = 0u;
        bool ok = false;
        while (!ok && polls < CHK_PROBE_MAX) {
          ++polls;
          __builtin_amdgcn_s_sleep(8);
          unsigned r = 0u;
          for (int w = off; w < off + 16 && start + w < end; ++w)
            r ^= chk_word(chk_ld_sys(ms + start + w), (unsigned)w);
          ok = (r ^ chk_seq(s)) == chk_ld32_sys(ck);
        }
        chk_probe_done(diag, polls, ok);
      }
    }
  }
}
#endif  // SABER_ARNQ_MAXG
