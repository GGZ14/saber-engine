// The null plugin: the two cheapest dispatches a gfx1201 queue can run.
//
// This is an INSTRUMENT, not a serve path. `saber chain-bench` measured 2.94 us of wall per
// dependent graph node beyond the kernel's own work and could not say what that number is made
// of, because its cheapest chain still ran a real kernel over real buffers. Three terms are
// folded into it:
//
//   (1) the command processor's per-dispatch work, plus the graph node's share of replay;
//   (2) the drain between two DEPENDENT dispatches -- the wait for the predecessor's writes to
//       be visible, which is a function of what the predecessor touched;
//   (3) the control kernel's own execution.
//
// Separating them needs a dispatch with no kernel work and no memory traffic, which is what
// `saber_noop` is, and one with a dependency of the smallest possible footprint, which is what
// `saber_noop_touch` is. The difference between chains of the two is (2) at one byte; the
// difference to the near-null control is (2)+(3) at 8 elements.
//
// Both are launched at grid 1 x block 1: one wave, one workgroup, one CU. Anything wider would
// measure occupancy ramp instead of dispatch.

// Zero arguments, zero memory traffic, zero instructions but the end-of-program. The kernarg
// segment holds only the HIP runtime's hidden block, so the host writes nothing at all.
extern "C" __global__ void saber_noop() {}

// One thread, one byte, read-modify-written. The load-store pair is what makes this kernel
// un-eliminable AND makes a chain of them a real RAW dependency: every node reads the byte the
// node before it wrote. `volatile` is deliberately NOT used -- the dependency has to be the
// hardware's, visible through the normal cache path, exactly as a real chain's is.
extern "C" __global__ void saber_noop_touch(unsigned char *p) {
    p[0] = (unsigned char)(p[0] + 1u);
}
