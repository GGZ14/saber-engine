// The DFlash2 selector's PATH WALK -- the kernel that turns the selector's edge scores into the
// draft block, and therefore the last thing `saber draft-step --proposal` has to reproduce.
//
// It IS a launch prod issues: the census has `_selector_walk_kernel` at 1 launch, 3.88 us, grid
// 32x1x1 at 32 threads -- one program per request, `num_warps=1`. `saber-model::draft` ports what
// it DECIDES; this ports what it RUNS.
//
// The reference is `dflash2/speculator.py:16`, and at prod's configuration two of its constexprs
// fall away:
//   SAMPLE_PROBABILISTIC   False -- `draft_sample_method: greedy`, so `self.draft_logits is None`
//   temperature            unused on that path, and `gumbel_noised_argmax` at `temp == 0` skips the
//                          division AND the noise, leaving `tl.max(logits, 0, return_indices=True)`
//                          -- a plain argmax whose ties go to the LOWEST index, Triton's default.
// What remains is a seven-step sequential walk:
//
//   previous = 0
//   for step in 0 .. num_steps:
//       flat  = row * num_steps + step
//       s[i]  = (i < top_k && valid) ? scores[(flat * top_k + previous) * top_k + i] : -inf
//       realized_scores[flat * top_k + i] = s[i]
//       index = argmax(s)                      ties to the lowest index
//       tokens[flat] = candidates[flat * top_k + index]      // both are int64
//       previous = index
//
// The sequential dependency through `previous` is the whole shape of it: step k+1 reads the row of
// the score matrix that step k's winner names, so this cannot be parallelised over steps and does
// not want to be -- 7 steps of 16 comparisons is one wave's work, which is why prod launches it
// with a single warp too.
#include <hip/hip_runtime.h>

#define DW_NT 32     // one wave, as the reference's num_warps=1

// Strictly greater wins; an exact tie goes to the LOWER index, which is what Triton's
// `tl.max(..., return_indices=True)` does. Order-independent, so the butterfly below is free.
__device__ __forceinline__ void dw_better(float &bv, int &bi, float v, int i) {
  const bool take = (v > bv) | ((v == bv) & (i < bi));
  bv = take ? v : bv;
  bi = take ? i : bi;
}

// THE CANDIDATE TABLE AND THE DRAFT BLOCK ARE `int64`, AND THAT IS PROD'S DTYPE AND NOT A CHOICE.
//
// `compute_candidates` returns `torch.topk`'s indices, which ATen types `int64`, and
// `_selector_walk_kernel` loads `candidate_ptr` and stores `tokens_ptr` at whatever dtype the
// tensors carry -- so both are eight bytes wide per element. This kernel read them as `int32`
// until 2026-09-21 and the failure was silent in exactly the way a width error is: every step's
// score row, every argmax and every realized score was prod's, bit for bit, and the tokens stored
// were `candidates[2i]`'s low half at even slots and zero at odd ones
// (`[1204, 1599, 30864, ...]` came out `[1204, 0, 1599, 314, 0, 0, 0]`).
//
// It survived every gate this kernel had because `capture_dflash_mlp.py` built its OWN candidate
// tensor at `int32` and compared `int32` tokens -- the capture and the kernel agreed on a width
// prod does not use. The capture now takes prod's dtypes, and `saber_model::draft::selector_walk`
// is the host reference with a synthetic table whose ODD slots hold distinct ids, which is the
// control an `int32` reader cannot pass.
extern "C" __global__ __launch_bounds__(DW_NT) void dflash_selector_walk_greedy(
    const float *__restrict__ scores,      // [reqs, steps, top_k, top_k] fp32
    const long *__restrict__ candidates,   // [reqs, steps, top_k] i64 -- prod's topk indices
    const int *__restrict__ req_state,     // [reqs, steps] i32; < 0 marks an inactive row
    long *__restrict__ tokens,             // [reqs, steps] i64, the draft block
    float *__restrict__ realized,          // [reqs, steps, top_k] fp32
    int reqs, int num_steps, int top_k) {
  const int row = (int)blockIdx.x;
  if (row >= reqs) return;
  const int lane = (int)threadIdx.x;
  // `req_state >= 0` is read ONCE, from step 0's slot, and gates every store -- the reference's
  // `valid`, which is a property of the request and not of the step.
  const bool valid = req_state[(long)row * num_steps] >= 0;
  int previous = 0;
  for (int step = 0; step < num_steps; ++step) {
    const long flat = (long)row * num_steps + step;
    const long base = (flat * top_k + previous) * top_k;
    const float s = (lane < top_k && valid) ? scores[base + lane] : -INFINITY;
    if (lane < top_k && valid) realized[flat * top_k + lane] = s;
    float bv = s;
    int bi = lane;
#pragma unroll
    for (int d = 1; d < DW_NT; d <<= 1) {
      const float ov = __shfl_xor(bv, d, DW_NT);
      const int oi = __shfl_xor(bi, d, DW_NT);
      dw_better(bv, bi, ov, oi);
    }
    if (lane == 0 && valid) tokens[flat] = candidates[flat * top_k + bi];
    previous = bi;
  }
}
