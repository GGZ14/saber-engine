// Saber source (CLEANROOM-1, 2026-10-25: frozen from a retired generator's output; edit this file directly)
//   ref/paro/par_kernels.h  sha256 9ef45b60288c38985fc759621f90e82cf62d9d3460380fdf65c48bb5fca44e0a
// Device code copied VERBATIM; only the six signature lines listed in the generator were
// rewritten, and the extern "C" wrappers at the end are new. No host code, no pybind, no getenv:
// every parameter the prod launcher chose from an environment variable or an `if` ladder is an
// explicit symbol here and a row in crates/saber-kernels/rules/paro_gfx1201.toml.
// Do not edit by hand: re-run the generator.
#include "par_kernels.h"

extern "C" __global__ __launch_bounds__(PQ_ROT_WAVES * 32)
void paro_add_rms_rot_rot0_i80(
    const __bf16 *__restrict__ Y, const __bf16 *__restrict__ RES,
    const __bf16 *__restrict__ Wn, float eps, const unsigned short *__restrict__ T,
    const __half *__restrict__ CS, __bf16 *__restrict__ HS, __bf16 *__restrict__ RO,
    unsigned char *__restrict__ A, float *__restrict__ ASG, float *__restrict__ RS,
    int M, int K, int krot, int gpw) {
  pq_add_rms_rot_impl<false, false>(
      Y, RES, Wn, eps, T, CS, HS, RO, A, ASG, RS, M, K, krot, gpw);
}

extern "C" __global__ __launch_bounds__(PQ_ROT_WAVES * 32)
void paro_add_rms_rot_rot1_i80(
    const __bf16 *__restrict__ Y, const __bf16 *__restrict__ RES,
    const __bf16 *__restrict__ Wn, float eps, const unsigned short *__restrict__ T,
    const __half *__restrict__ CS, __bf16 *__restrict__ HS, __bf16 *__restrict__ RO,
    unsigned char *__restrict__ A, float *__restrict__ ASG, float *__restrict__ RS,
    int M, int K, int krot, int gpw) {
  pq_add_rms_rot_impl<true, false>(
      Y, RES, Wn, eps, T, CS, HS, RO, A, ASG, RS, M, K, krot, gpw);
}

extern "C" __global__ __launch_bounds__(PQ_ROT_WAVES * 32)
void paro_add_rms_rot_rot1_i81(
    const __bf16 *__restrict__ Y, const __bf16 *__restrict__ RES,
    const __bf16 *__restrict__ Wn, float eps, const unsigned short *__restrict__ T,
    const __half *__restrict__ CS, __bf16 *__restrict__ HS, __bf16 *__restrict__ RO,
    unsigned char *__restrict__ A, float *__restrict__ ASG, float *__restrict__ RS,
    int M, int K, int krot, int gpw) {
  pq_add_rms_rot_impl<true, true>(
      Y, RES, Wn, eps, T, CS, HS, RO, A, ASG, RS, M, K, krot, gpw);
}

extern "C" __global__ __launch_bounds__(PQ_ROT_WAVES * 32)
void paro_ew_rot_m0_rot0_i80(
    const __bf16 *__restrict__ X, const __bf16 *__restrict__ Y, long ys,
    const __bf16 *__restrict__ Wn, float eps, const unsigned short *__restrict__ T,
    const __half *__restrict__ CS, __bf16 *__restrict__ HS, unsigned char *__restrict__ A,
    float *__restrict__ ASG, float *__restrict__ RS, int M, int N, int krot, int gpw) {
  pq_ew_rot_impl<0, false, false>(
      X, Y, ys, Wn, eps, T, CS, HS, A, ASG, RS, M, N, krot, gpw);
}

extern "C" __global__ __launch_bounds__(PQ_ROT_WAVES * 32)
void paro_ew_rot_m0_rot1_i80(
    const __bf16 *__restrict__ X, const __bf16 *__restrict__ Y, long ys,
    const __bf16 *__restrict__ Wn, float eps, const unsigned short *__restrict__ T,
    const __half *__restrict__ CS, __bf16 *__restrict__ HS, unsigned char *__restrict__ A,
    float *__restrict__ ASG, float *__restrict__ RS, int M, int N, int krot, int gpw) {
  pq_ew_rot_impl<0, true, false>(
      X, Y, ys, Wn, eps, T, CS, HS, A, ASG, RS, M, N, krot, gpw);
}

extern "C" __global__ __launch_bounds__(PQ_ROT_WAVES * 32)
void paro_ew_rot_m0_rot1_i81(
    const __bf16 *__restrict__ X, const __bf16 *__restrict__ Y, long ys,
    const __bf16 *__restrict__ Wn, float eps, const unsigned short *__restrict__ T,
    const __half *__restrict__ CS, __bf16 *__restrict__ HS, unsigned char *__restrict__ A,
    float *__restrict__ ASG, float *__restrict__ RS, int M, int N, int krot, int gpw) {
  pq_ew_rot_impl<0, true, true>(
      X, Y, ys, Wn, eps, T, CS, HS, A, ASG, RS, M, N, krot, gpw);
}

extern "C" __global__ __launch_bounds__(PQ_ROT_WAVES * 32)
void paro_ew_rot_m1_rot0_i80(
    const __bf16 *__restrict__ X, const __bf16 *__restrict__ Y, long ys,
    const __bf16 *__restrict__ Wn, float eps, const unsigned short *__restrict__ T,
    const __half *__restrict__ CS, __bf16 *__restrict__ HS, unsigned char *__restrict__ A,
    float *__restrict__ ASG, float *__restrict__ RS, int M, int N, int krot, int gpw) {
  pq_ew_rot_impl<1, false, false>(
      X, Y, ys, Wn, eps, T, CS, HS, A, ASG, RS, M, N, krot, gpw);
}

extern "C" __global__ __launch_bounds__(PQ_ROT_WAVES * 32)
void paro_ew_rot_m1_rot1_i80(
    const __bf16 *__restrict__ X, const __bf16 *__restrict__ Y, long ys,
    const __bf16 *__restrict__ Wn, float eps, const unsigned short *__restrict__ T,
    const __half *__restrict__ CS, __bf16 *__restrict__ HS, unsigned char *__restrict__ A,
    float *__restrict__ ASG, float *__restrict__ RS, int M, int N, int krot, int gpw) {
  pq_ew_rot_impl<1, true, false>(
      X, Y, ys, Wn, eps, T, CS, HS, A, ASG, RS, M, N, krot, gpw);
}

extern "C" __global__ __launch_bounds__(PQ_ROT_WAVES * 32)
void paro_ew_rot_m1_rot1_i81(
    const __bf16 *__restrict__ X, const __bf16 *__restrict__ Y, long ys,
    const __bf16 *__restrict__ Wn, float eps, const unsigned short *__restrict__ T,
    const __half *__restrict__ CS, __bf16 *__restrict__ HS, unsigned char *__restrict__ A,
    float *__restrict__ ASG, float *__restrict__ RS, int M, int N, int krot, int gpw) {
  pq_ew_rot_impl<1, true, true>(
      X, Y, ys, Wn, eps, T, CS, HS, A, ASG, RS, M, N, krot, gpw);
}

extern "C" __global__ __launch_bounds__(PQ_ROT_WAVES * 32)
void paro_ew_rot_m2_rot0_i80(
    const __bf16 *__restrict__ X, const __bf16 *__restrict__ Y, long ys,
    const __bf16 *__restrict__ Wn, float eps, const unsigned short *__restrict__ T,
    const __half *__restrict__ CS, __bf16 *__restrict__ HS, unsigned char *__restrict__ A,
    float *__restrict__ ASG, float *__restrict__ RS, int M, int N, int krot, int gpw) {
  pq_ew_rot_impl<2, false, false>(
      X, Y, ys, Wn, eps, T, CS, HS, A, ASG, RS, M, N, krot, gpw);
}

extern "C" __global__ __launch_bounds__(PQ_ROT_WAVES * 32)
void paro_ew_rot_m2_rot1_i80(
    const __bf16 *__restrict__ X, const __bf16 *__restrict__ Y, long ys,
    const __bf16 *__restrict__ Wn, float eps, const unsigned short *__restrict__ T,
    const __half *__restrict__ CS, __bf16 *__restrict__ HS, unsigned char *__restrict__ A,
    float *__restrict__ ASG, float *__restrict__ RS, int M, int N, int krot, int gpw) {
  pq_ew_rot_impl<2, true, false>(
      X, Y, ys, Wn, eps, T, CS, HS, A, ASG, RS, M, N, krot, gpw);
}

extern "C" __global__ __launch_bounds__(PQ_ROT_WAVES * 32)
void paro_ew_rot_m2_rot1_i81(
    const __bf16 *__restrict__ X, const __bf16 *__restrict__ Y, long ys,
    const __bf16 *__restrict__ Wn, float eps, const unsigned short *__restrict__ T,
    const __half *__restrict__ CS, __bf16 *__restrict__ HS, unsigned char *__restrict__ A,
    float *__restrict__ ASG, float *__restrict__ RS, int M, int N, int krot, int gpw) {
  pq_ew_rot_impl<2, true, true>(
      X, Y, ys, Wn, eps, T, CS, HS, A, ASG, RS, M, N, krot, gpw);
}

extern "C" __global__ __launch_bounds__(8 * 32)
void paro_add_rms_rot_tok_w8_tiled0(
    const __bf16 *__restrict__ Y, const __bf16 *__restrict__ RES,
    const __bf16 *__restrict__ Wn, float eps, const unsigned short *__restrict__ T,
    const __half *__restrict__ CS, __bf16 *__restrict__ HS, __bf16 *__restrict__ RO,
    unsigned char *__restrict__ A, float *__restrict__ AS, int M, int K, int krot) {
  pq_add_rms_rot_tok_impl<8, false, false>(
      Y, RES, Wn, eps, T, CS, HS, RO, A, AS, M, K, krot);
}

extern "C" __global__ __launch_bounds__(8 * 32)
void paro_add_rms_rot_tok_w8_tiled1(
    const __bf16 *__restrict__ Y, const __bf16 *__restrict__ RES,
    const __bf16 *__restrict__ Wn, float eps, const unsigned short *__restrict__ T,
    const __half *__restrict__ CS, __bf16 *__restrict__ HS, __bf16 *__restrict__ RO,
    unsigned char *__restrict__ A, float *__restrict__ AS, int M, int K, int krot) {
  pq_add_rms_rot_tok_impl<8, true, false>(
      Y, RES, Wn, eps, T, CS, HS, RO, A, AS, M, K, krot);
}

extern "C" __global__ __launch_bounds__(16 * 32)
void paro_add_rms_rot_tok_w16_tiled0(
    const __bf16 *__restrict__ Y, const __bf16 *__restrict__ RES,
    const __bf16 *__restrict__ Wn, float eps, const unsigned short *__restrict__ T,
    const __half *__restrict__ CS, __bf16 *__restrict__ HS, __bf16 *__restrict__ RO,
    unsigned char *__restrict__ A, float *__restrict__ AS, int M, int K, int krot) {
  pq_add_rms_rot_tok_impl<16, false, false>(
      Y, RES, Wn, eps, T, CS, HS, RO, A, AS, M, K, krot);
}

extern "C" __global__ __launch_bounds__(16 * 32)
void paro_add_rms_rot_tok_w16_tiled1(
    const __bf16 *__restrict__ Y, const __bf16 *__restrict__ RES,
    const __bf16 *__restrict__ Wn, float eps, const unsigned short *__restrict__ T,
    const __half *__restrict__ CS, __bf16 *__restrict__ HS, __bf16 *__restrict__ RO,
    unsigned char *__restrict__ A, float *__restrict__ AS, int M, int K, int krot) {
  pq_add_rms_rot_tok_impl<16, true, false>(
      Y, RES, Wn, eps, T, CS, HS, RO, A, AS, M, K, krot);
}

extern "C" __global__ __launch_bounds__(32 * 32)
void paro_add_rms_rot_tok_w32_tiled0(
    const __bf16 *__restrict__ Y, const __bf16 *__restrict__ RES,
    const __bf16 *__restrict__ Wn, float eps, const unsigned short *__restrict__ T,
    const __half *__restrict__ CS, __bf16 *__restrict__ HS, __bf16 *__restrict__ RO,
    unsigned char *__restrict__ A, float *__restrict__ AS, int M, int K, int krot) {
  pq_add_rms_rot_tok_impl<32, false, false>(
      Y, RES, Wn, eps, T, CS, HS, RO, A, AS, M, K, krot);
}

extern "C" __global__ __launch_bounds__(32 * 32)
void paro_add_rms_rot_tok_w32_tiled1(
    const __bf16 *__restrict__ Y, const __bf16 *__restrict__ RES,
    const __bf16 *__restrict__ Wn, float eps, const unsigned short *__restrict__ T,
    const __half *__restrict__ CS, __bf16 *__restrict__ HS, __bf16 *__restrict__ RO,
    unsigned char *__restrict__ A, float *__restrict__ AS, int M, int K, int krot) {
  pq_add_rms_rot_tok_impl<32, true, false>(
      Y, RES, Wn, eps, T, CS, HS, RO, A, AS, M, K, krot);
}

extern "C" __global__ __launch_bounds__(8 * 32)
void paro_ew_rot_tok_m0_w8_tiled0(
    const __bf16 *__restrict__ X, const __bf16 *__restrict__ Y, long ys,
    const __bf16 *__restrict__ Wn, float eps, const unsigned short *__restrict__ T,
    const __half *__restrict__ CS, __bf16 *__restrict__ HS, unsigned char *__restrict__ A,
    float *__restrict__ AS, int M, int N, int krot) {
  pq_ew_rot_tok_impl<0, 8, false, false>(
      X, Y, ys, Wn, eps, T, CS, HS, A, AS, M, N, krot);
}

extern "C" __global__ __launch_bounds__(8 * 32)
void paro_ew_rot_tok_m0_w8_tiled1(
    const __bf16 *__restrict__ X, const __bf16 *__restrict__ Y, long ys,
    const __bf16 *__restrict__ Wn, float eps, const unsigned short *__restrict__ T,
    const __half *__restrict__ CS, __bf16 *__restrict__ HS, unsigned char *__restrict__ A,
    float *__restrict__ AS, int M, int N, int krot) {
  pq_ew_rot_tok_impl<0, 8, true, false>(
      X, Y, ys, Wn, eps, T, CS, HS, A, AS, M, N, krot);
}

extern "C" __global__ __launch_bounds__(16 * 32)
void paro_ew_rot_tok_m0_w16_tiled0(
    const __bf16 *__restrict__ X, const __bf16 *__restrict__ Y, long ys,
    const __bf16 *__restrict__ Wn, float eps, const unsigned short *__restrict__ T,
    const __half *__restrict__ CS, __bf16 *__restrict__ HS, unsigned char *__restrict__ A,
    float *__restrict__ AS, int M, int N, int krot) {
  pq_ew_rot_tok_impl<0, 16, false, false>(
      X, Y, ys, Wn, eps, T, CS, HS, A, AS, M, N, krot);
}

extern "C" __global__ __launch_bounds__(16 * 32)
void paro_ew_rot_tok_m0_w16_tiled1(
    const __bf16 *__restrict__ X, const __bf16 *__restrict__ Y, long ys,
    const __bf16 *__restrict__ Wn, float eps, const unsigned short *__restrict__ T,
    const __half *__restrict__ CS, __bf16 *__restrict__ HS, unsigned char *__restrict__ A,
    float *__restrict__ AS, int M, int N, int krot) {
  pq_ew_rot_tok_impl<0, 16, true, false>(
      X, Y, ys, Wn, eps, T, CS, HS, A, AS, M, N, krot);
}

extern "C" __global__ __launch_bounds__(32 * 32)
void paro_ew_rot_tok_m0_w32_tiled0(
    const __bf16 *__restrict__ X, const __bf16 *__restrict__ Y, long ys,
    const __bf16 *__restrict__ Wn, float eps, const unsigned short *__restrict__ T,
    const __half *__restrict__ CS, __bf16 *__restrict__ HS, unsigned char *__restrict__ A,
    float *__restrict__ AS, int M, int N, int krot) {
  pq_ew_rot_tok_impl<0, 32, false, false>(
      X, Y, ys, Wn, eps, T, CS, HS, A, AS, M, N, krot);
}

extern "C" __global__ __launch_bounds__(32 * 32)
void paro_ew_rot_tok_m0_w32_tiled1(
    const __bf16 *__restrict__ X, const __bf16 *__restrict__ Y, long ys,
    const __bf16 *__restrict__ Wn, float eps, const unsigned short *__restrict__ T,
    const __half *__restrict__ CS, __bf16 *__restrict__ HS, unsigned char *__restrict__ A,
    float *__restrict__ AS, int M, int N, int krot) {
  pq_ew_rot_tok_impl<0, 32, true, false>(
      X, Y, ys, Wn, eps, T, CS, HS, A, AS, M, N, krot);
}

extern "C" __global__ __launch_bounds__(8 * 32)
void paro_ew_rot_tok_m1_w8_tiled0(
    const __bf16 *__restrict__ X, const __bf16 *__restrict__ Y, long ys,
    const __bf16 *__restrict__ Wn, float eps, const unsigned short *__restrict__ T,
    const __half *__restrict__ CS, __bf16 *__restrict__ HS, unsigned char *__restrict__ A,
    float *__restrict__ AS, int M, int N, int krot) {
  pq_ew_rot_tok_impl<1, 8, false, false>(
      X, Y, ys, Wn, eps, T, CS, HS, A, AS, M, N, krot);
}

extern "C" __global__ __launch_bounds__(8 * 32)
void paro_ew_rot_tok_m1_w8_tiled1(
    const __bf16 *__restrict__ X, const __bf16 *__restrict__ Y, long ys,
    const __bf16 *__restrict__ Wn, float eps, const unsigned short *__restrict__ T,
    const __half *__restrict__ CS, __bf16 *__restrict__ HS, unsigned char *__restrict__ A,
    float *__restrict__ AS, int M, int N, int krot) {
  pq_ew_rot_tok_impl<1, 8, true, false>(
      X, Y, ys, Wn, eps, T, CS, HS, A, AS, M, N, krot);
}

extern "C" __global__ __launch_bounds__(16 * 32)
void paro_ew_rot_tok_m1_w16_tiled0(
    const __bf16 *__restrict__ X, const __bf16 *__restrict__ Y, long ys,
    const __bf16 *__restrict__ Wn, float eps, const unsigned short *__restrict__ T,
    const __half *__restrict__ CS, __bf16 *__restrict__ HS, unsigned char *__restrict__ A,
    float *__restrict__ AS, int M, int N, int krot) {
  pq_ew_rot_tok_impl<1, 16, false, false>(
      X, Y, ys, Wn, eps, T, CS, HS, A, AS, M, N, krot);
}

extern "C" __global__ __launch_bounds__(16 * 32)
void paro_ew_rot_tok_m1_w16_tiled1(
    const __bf16 *__restrict__ X, const __bf16 *__restrict__ Y, long ys,
    const __bf16 *__restrict__ Wn, float eps, const unsigned short *__restrict__ T,
    const __half *__restrict__ CS, __bf16 *__restrict__ HS, unsigned char *__restrict__ A,
    float *__restrict__ AS, int M, int N, int krot) {
  pq_ew_rot_tok_impl<1, 16, true, false>(
      X, Y, ys, Wn, eps, T, CS, HS, A, AS, M, N, krot);
}

extern "C" __global__ __launch_bounds__(32 * 32)
void paro_ew_rot_tok_m1_w32_tiled0(
    const __bf16 *__restrict__ X, const __bf16 *__restrict__ Y, long ys,
    const __bf16 *__restrict__ Wn, float eps, const unsigned short *__restrict__ T,
    const __half *__restrict__ CS, __bf16 *__restrict__ HS, unsigned char *__restrict__ A,
    float *__restrict__ AS, int M, int N, int krot) {
  pq_ew_rot_tok_impl<1, 32, false, false>(
      X, Y, ys, Wn, eps, T, CS, HS, A, AS, M, N, krot);
}

extern "C" __global__ __launch_bounds__(32 * 32)
void paro_ew_rot_tok_m1_w32_tiled1(
    const __bf16 *__restrict__ X, const __bf16 *__restrict__ Y, long ys,
    const __bf16 *__restrict__ Wn, float eps, const unsigned short *__restrict__ T,
    const __half *__restrict__ CS, __bf16 *__restrict__ HS, unsigned char *__restrict__ A,
    float *__restrict__ AS, int M, int N, int krot) {
  pq_ew_rot_tok_impl<1, 32, true, false>(
      X, Y, ys, Wn, eps, T, CS, HS, A, AS, M, N, krot);
}

extern "C" __global__ __launch_bounds__(8 * 32)
void paro_ew_rot_tok_m2_w8_tiled0(
    const __bf16 *__restrict__ X, const __bf16 *__restrict__ Y, long ys,
    const __bf16 *__restrict__ Wn, float eps, const unsigned short *__restrict__ T,
    const __half *__restrict__ CS, __bf16 *__restrict__ HS, unsigned char *__restrict__ A,
    float *__restrict__ AS, int M, int N, int krot) {
  pq_ew_rot_tok_impl<2, 8, false, false>(
      X, Y, ys, Wn, eps, T, CS, HS, A, AS, M, N, krot);
}

extern "C" __global__ __launch_bounds__(8 * 32)
void paro_ew_rot_tok_m2_w8_tiled1(
    const __bf16 *__restrict__ X, const __bf16 *__restrict__ Y, long ys,
    const __bf16 *__restrict__ Wn, float eps, const unsigned short *__restrict__ T,
    const __half *__restrict__ CS, __bf16 *__restrict__ HS, unsigned char *__restrict__ A,
    float *__restrict__ AS, int M, int N, int krot) {
  pq_ew_rot_tok_impl<2, 8, true, false>(
      X, Y, ys, Wn, eps, T, CS, HS, A, AS, M, N, krot);
}

extern "C" __global__ __launch_bounds__(16 * 32)
void paro_ew_rot_tok_m2_w16_tiled0(
    const __bf16 *__restrict__ X, const __bf16 *__restrict__ Y, long ys,
    const __bf16 *__restrict__ Wn, float eps, const unsigned short *__restrict__ T,
    const __half *__restrict__ CS, __bf16 *__restrict__ HS, unsigned char *__restrict__ A,
    float *__restrict__ AS, int M, int N, int krot) {
  pq_ew_rot_tok_impl<2, 16, false, false>(
      X, Y, ys, Wn, eps, T, CS, HS, A, AS, M, N, krot);
}

extern "C" __global__ __launch_bounds__(16 * 32)
void paro_ew_rot_tok_m2_w16_tiled1(
    const __bf16 *__restrict__ X, const __bf16 *__restrict__ Y, long ys,
    const __bf16 *__restrict__ Wn, float eps, const unsigned short *__restrict__ T,
    const __half *__restrict__ CS, __bf16 *__restrict__ HS, unsigned char *__restrict__ A,
    float *__restrict__ AS, int M, int N, int krot) {
  pq_ew_rot_tok_impl<2, 16, true, false>(
      X, Y, ys, Wn, eps, T, CS, HS, A, AS, M, N, krot);
}

extern "C" __global__ __launch_bounds__(32 * 32)
void paro_ew_rot_tok_m2_w32_tiled0(
    const __bf16 *__restrict__ X, const __bf16 *__restrict__ Y, long ys,
    const __bf16 *__restrict__ Wn, float eps, const unsigned short *__restrict__ T,
    const __half *__restrict__ CS, __bf16 *__restrict__ HS, unsigned char *__restrict__ A,
    float *__restrict__ AS, int M, int N, int krot) {
  pq_ew_rot_tok_impl<2, 32, false, false>(
      X, Y, ys, Wn, eps, T, CS, HS, A, AS, M, N, krot);
}

extern "C" __global__ __launch_bounds__(32 * 32)
void paro_ew_rot_tok_m2_w32_tiled1(
    const __bf16 *__restrict__ X, const __bf16 *__restrict__ Y, long ys,
    const __bf16 *__restrict__ Wn, float eps, const unsigned short *__restrict__ T,
    const __half *__restrict__ CS, __bf16 *__restrict__ HS, unsigned char *__restrict__ A,
    float *__restrict__ AS, int M, int N, int krot) {
  pq_ew_rot_tok_impl<2, 32, true, false>(
      X, Y, ys, Wn, eps, T, CS, HS, A, AS, M, N, krot);
}
