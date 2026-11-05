// FN71: NVMe -> VRAM through the amdgpu P2PDMA pool (kernel patch 0001, amdgpu.p2pmem_mb > 0), ONE
// card (HIP_VISIBLE_DEVICES=1).
//
//   fn71_verify FILE OUTDIR POOLFILE
//
// POOLFILE = the card's /sys/bus/pci/devices/<bdf>/p2pmem/allocate, bind-mounted rw into the
// container (sysfs itself is read-only there). Phases (all into OUTDIR/verify.txt):
//  A  the pool: sysfs size/available/published; the render node; GEM_CREATE(P2PMEM_LOCAL) -> handle
//     (+ a wrong-size call must be EINVAL) -> PRIME fd -> GPU pointers by hipImportExternalMemory
//     and by hsa_amd_interop_map_buffer (saber-hsa's path); mmap of the WHOLE pool (MAP_SHARED).
//  B  aliasing: the GPU writes a pattern, the CPU reads it through the pool mapping (offset x == x);
//     the CPU writes another, the GPU reads it back (HIP and HSA pointers).
//  C  the direct read: pread(O_DIRECT) of one 2.62 MB record into the pool mapping (errno if not).
//  D  correctness: min(1 GiB, pool) of FILE from offset 0 into the pool (8 MiB preads, QD 8); a HIP
//     kernel copies the pool to a hipMalloc buffer; sha256 of the readback vs sha256 of the same
//     bytes read into host memory; memcmp.
//  E  the rewrite law: 4 MiB the GPU just read is overwritten by the drive with other bytes and read
//     again by a fresh kernel -- EQUAL, or STALE (a GPU cache kept the old lines).
//  F  rates (FN70's harness, 2.62 MB records, QD 1/8/32, 1024 records a point, + a 1 GiB stream of
//     8 MiB pieces at QD 8): drive->pinned, the bounce (drive->pinned->H2D), P2P (drive->pool).
//  G  host DRAM contention: 8 CPU threads stream-read 2 GiB alone and beside each form at QD 32.
// Build (dev image): hipcc -O2 -std=c++17 fn71_verify.cpp -o fn71_verify -lhsa-runtime64
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <dirent.h>
#include <fcntl.h>
#include <limits.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#define CK(x)                                                                                    \
  do {                                                                                           \
    hipError_t e_ = (x);                                                                         \
    if (e_ != hipSuccess) {                                                                      \
      fprintf(stderr, "HIP %s at %s:%d: %s\n", hipGetErrorString(e_), __FILE__, __LINE__, #x);   \
      exit(2);                                                                                   \
    }                                                                                            \
  } while (0)

// ---- the two ioctls (no libdrm headers in the image; layouts from include/uapi/drm)
struct GemCreateIn {
  uint64_t bo_size, alignment, domains, domain_flags;
};
union GemCreate {
  GemCreateIn in;
  struct {
    uint32_t handle, pad;
  } out;
};
struct PrimeHandle {
  uint32_t handle, flags;
  int32_t fd;
};
#define FN71_GEM_CREATE _IOWR('d', 0x40 + 0x00, union GemCreate)
#define FN71_PRIME_HANDLE_TO_FD _IOWR('d', 0x2d, struct PrimeHandle)
static const uint64_t P2PMEM_LOCAL = 1ull << 48;  // AMDGPU_GEM_CREATE_P2PMEM_LOCAL (patch 0001)
static const uint64_t DOMAIN_VRAM = 0x4;

static const size_t REC = 2621440;  // one expert record (2.62 MB, 4096-aligned)
static const size_t MiB = 1 << 20;
static double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static FILE *g_out = nullptr;
static void say(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void say(const char *fmt, ...) {
  va_list a, b;
  va_start(a, fmt);
  va_copy(b, a);
  vprintf(fmt, a);
  if (g_out) vfprintf(g_out, fmt, b);
  va_end(a);
  va_end(b);
  fflush(stdout);
  if (g_out) fflush(g_out);
}

// ---- sha256 (FIPS 180-4), for the readback-vs-file equality line
struct Sha256 {
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  uint8_t buf[64];
  size_t n = 0;
  uint64_t len = 0;
  static uint32_t r(uint32_t x, int s) { return (x >> s) | (x << (32 - s)); }
  void block(const uint8_t *p) {
    static const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
        0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
        0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
        0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) w[i] = (p[4 * i] << 24) | (p[4 * i + 1] << 16) | (p[4 * i + 2] << 8) | p[4 * i + 3];
    for (int i = 16; i < 64; ++i)
      w[i] = w[i - 16] + (r(w[i - 15], 7) ^ r(w[i - 15], 18) ^ (w[i - 15] >> 3)) + w[i - 7] +
             (r(w[i - 2], 17) ^ r(w[i - 2], 19) ^ (w[i - 2] >> 10));
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
      uint32_t t1 = hh + (r(e, 6) ^ r(e, 11) ^ r(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
      uint32_t t2 = (r(a, 2) ^ r(a, 13) ^ r(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
      hh = g, g = f, f = e, e = d + t1, d = c, c = b, b = a, a = t1 + t2;
    }
    h[0] += a, h[1] += b, h[2] += c, h[3] += d, h[4] += e, h[5] += f, h[6] += g, h[7] += hh;
  }
  void add(const uint8_t *p, size_t m) {
    len += m;
    while (m) {
      size_t t = std::min(m, 64 - n);
      memcpy(buf + n, p, t);
      n += t, p += t, m -= t;
      if (n == 64) block(buf), n = 0;
    }
  }
  std::string hex() {
    uint64_t bits = len * 8;
    uint8_t pad = 0x80, z = 0;
    add(&pad, 1);
    while (n != 56) add(&z, 1);
    for (int i = 7; i >= 0; --i) {
      uint8_t c = bits >> (8 * i);
      add(&c, 1);
    }
    char s[65];
    for (int i = 0; i < 8; ++i) snprintf(s + 8 * i, 9, "%08x", h[i]);
    return s;
  }
};
static std::string sha256(const void *p, size_t n) {
  Sha256 s;
  s.add((const uint8_t *)p, n);
  return s.hex();
}

// ---- kernels
__global__ void k_copy(const uint4 *__restrict__ s, uint4 *__restrict__ d, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) d[i] = s[i];
}
__device__ __host__ inline uint32_t pat(uint64_t i, uint32_t seed) {
  uint64_t x = (i + 1) * 0x9E3779B97F4A7C15ull ^ seed;
  x ^= x >> 31;
  return (uint32_t)(x * 0xBF58476D1CE4E5B9ull >> 32);
}
__global__ void k_fill(uint32_t *d, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    d[i] = pat(i, seed);
}
// FN71b: the in-kernel rewrite probe. One workgroup reads `region` (warming L0/GL1/GL2), raises
// `ack`, waits for the host's `flag` (the completion word written after the drive's pread returned),
// then reads `region` again into out2. mode 0: volatile spin, plain loads (no fence); mode 1: volatile
// spin + __threadfence_system() in every thread; mode 2: every thread's own acquire-system load of the
// flag (the HIP memory model's form of "the completion word is the release").
__global__ void k_spin_reread(const uint4 *region, uint4 *out1, uint4 *out2, size_t n, uint32_t *flag,
                              uint32_t *ack, int mode) {
  for (size_t i = threadIdx.x; i < n; i += blockDim.x) out1[i] = region[i];
  __syncthreads();
  if (threadIdx.x == 0) {
    __hip_atomic_store(ack, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    while (*(volatile uint32_t *)flag == 0) {
    }
  }
  __syncthreads();
  if (mode == 1) __threadfence_system();
  if (mode == 2)
    while (__hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) == 0) {
    }
  for (size_t i = threadIdx.x; i < n; i += blockDim.x) out2[i] = region[i];
}
// FN71b latency: thread 0 waits for the completion word (acquire, system) when `flag` is set, reads
// the record's first and last 16 bytes, writes them + raises ack (release, system).
__global__ void k_first_byte(const uint4 *rec, size_t n, uint32_t *flag, uint32_t *ack, uint4 *out) {
  if (threadIdx.x) return;
  if (flag)
    while (__hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) == 0) {
    }
  uint4 a = rec[0], b = rec[n - 1];
  out[0] = a, out[1] = b;
  __hip_atomic_store(ack, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
static void gpu_copy(const void *src, void *dst, size_t bytes) {
  k_copy<<<1024, 256>>>((const uint4 *)src, (uint4 *)dst, bytes / 16);
  CK(hipGetLastError());
  CK(hipDeviceSynchronize());
}
// GPU reads `bytes` at `dev` into a host buffer (through a hipMalloc bounce the kernel writes)
static void gpu_readback(const void *dev, void *host, size_t bytes, void *scratch) {
  gpu_copy(dev, scratch, bytes);
  CK(hipMemcpy(host, scratch, bytes, hipMemcpyDeviceToHost));
}

// ---- HSA agent for the card
static hsa_agent_t g_gpu = {0};
static uint32_t g_want_bdf = 0;
static hsa_status_t agent_cb(hsa_agent_t a, void *) {
  hsa_device_type_t t;
  hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t);
  if (t == HSA_DEVICE_TYPE_GPU) {
    uint32_t bdf = 0;
    hsa_agent_get_info(a, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID, &bdf);
    if (bdf == g_want_bdf) g_gpu = a;
  }
  return HSA_STATUS_SUCCESS;
}

static std::string slurp(const std::string &p) {
  FILE *f = fopen(p.c_str(), "r");
  if (!f) return std::string("<") + strerror(errno) + ">";
  char b[256] = {0};
  size_t n = fread(b, 1, sizeof(b) - 1, f);
  fclose(f);
  while (n && (b[n - 1] == '\n' || b[n - 1] == ' ')) b[--n] = 0;
  return b;
}
static std::string render_node_for(const std::string &bdf) {
  DIR *d = opendir("/sys/class/drm");
  if (!d) return "";
  std::string hit;
  while (dirent *e = readdir(d)) {
    if (strncmp(e->d_name, "renderD", 7)) continue;
    char rp[PATH_MAX];
    std::string l = std::string("/sys/class/drm/") + e->d_name + "/device";
    if (!realpath(l.c_str(), rp)) continue;
    const char *b = strrchr(rp, '/');
    if (b && bdf == b + 1) hit = std::string("/dev/dri/") + e->d_name;
  }
  closedir(d);
  return hit;
}

// ---- transfer forms (FN70's harness)
static int g_fd_direct = -1;
static std::atomic<bool> g_dram_run{false};
static std::atomic<uint64_t> g_dram_bytes{0};
static void dram_thread(const uint64_t *p, size_t n) {
  uint64_t acc = 0;
  while (g_dram_run.load(std::memory_order_relaxed)) {
    for (size_t i = 0; i < n; i += 8)
      acc += p[i] ^ p[i + 1] ^ p[i + 2] ^ p[i + 3] ^ p[i + 4] ^ p[i + 5] ^ p[i + 6] ^ p[i + 7];
    g_dram_bytes.fetch_add(n * 8, std::memory_order_relaxed);
  }
  if (acc == 42) printf(" ");
}
enum Form { DRIVE_ONLY, BOUNCE, P2P };
static const char *form_name(Form f) {
  return f == DRIVE_ONLY ? "drive->pinned" : f == BOUNCE ? "bounce(drive->pinned->H2D)" : "p2p(drive->pool VRAM)";
}
struct Lat {
  std::vector<double> v;
  double pct(double q) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, (size_t)(q * v.size()))];
  }
};
static double run_form(Form f, int qd, size_t piece, size_t n, uint64_t span, char *vram, char *pool, size_t pool_sz,
                       Lat *lat, double min_secs = 0) {
  std::atomic<size_t> next{0};
  std::vector<std::thread> th;
  std::vector<Lat> lats(qd);
  std::atomic<int> bad{0};
  double t0 = now_s();
  for (int t = 0; t < qd; ++t)
    th.emplace_back([&, t]() {
      CK(hipSetDevice(0));
      void *pin = nullptr;
      if (f != P2P) CK(hipHostMalloc(&pin, piece, hipHostMallocDefault));
      hipStream_t st;
      CK(hipStreamCreateWithFlags(&st, hipStreamNonBlocking));
      for (;;) {
        size_t k = next.fetch_add(1);
        if (k >= n) {
          if (min_secs > 0 && now_s() - t0 < min_secs)
            k = k % n;
          else
            break;
        }
        uint64_t off = ((k * 0x9E3779B97F4A7C15ull) % (span / piece)) * piece;
        size_t vslot = (size_t)t * piece % (256 * MiB - piece + 1);
        size_t pslot = ((size_t)t * piece) % (pool_sz - piece + 1) / 4096 * 4096;
        double a = now_s();
        char *dst = f == P2P ? pool + pslot : (char *)pin;
        size_t got = 0;
        while (got < piece) {
          ssize_t r = pread(g_fd_direct, dst + got, piece - got, off + got);
          if (r <= 0) {
            bad++;
            break;
          }
          got += r;
        }
        if (f == BOUNCE) {
          CK(hipMemcpyAsync(vram + vslot, pin, piece, hipMemcpyHostToDevice, st));
          CK(hipStreamSynchronize(st));
        }
        lats[t].v.push_back((now_s() - a) * 1e6);
      }
      CK(hipStreamDestroy(st));
      if (pin) CK(hipHostFree(pin));
    });
  for (auto &x : th) x.join();
  double dt = now_s() - t0;
  size_t done = 0;
  for (auto &l : lats) {
    done += l.v.size();
    if (lat) lat->v.insert(lat->v.end(), l.v.begin(), l.v.end());
  }
  if (bad) say("  !! %d short/failed reads\n", bad.load());
  return done * (double)piece / dt / 1e9;
}

// QD-8 8 MiB-piece read of [foff, foff+bytes) into dst (host-visible: pinned RAM or the pool)
static bool read_span(char *dst, uint64_t foff, size_t bytes) {
  std::atomic<size_t> next{0};
  std::atomic<int> bad{0};
  const size_t P = 8 * MiB;
  std::vector<std::thread> th;
  for (int t = 0; t < 8; ++t)
    th.emplace_back([&]() {
      for (;;) {
        size_t k = next.fetch_add(1);
        if (k * P >= bytes) break;
        size_t len = std::min(P, bytes - k * P);
        if (pread(g_fd_direct, dst + k * P, len, foff + k * P) != (ssize_t)len) bad++;
      }
    });
  for (auto &x : th) x.join();
  return bad == 0;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: fn71_verify FILE OUTDIR POOLFILE\n");
    return 1;
  }
  g_out = fopen((std::string(argv[2]) + "/verify.txt").c_str(), "w");
  g_fd_direct = open(argv[1], O_RDONLY | O_DIRECT);
  if (g_fd_direct < 0) {
    perror("open FILE");
    return 1;
  }
  off_t fsz = lseek(g_fd_direct, 0, SEEK_END);
  CK(hipSetDevice(0));
  hipDeviceProp_t pr;
  CK(hipGetDeviceProperties(&pr, 0));
  char bdf[32];
  snprintf(bdf, sizeof bdf, "%04x:%02x:%02x.0", pr.pciDomainID, pr.pciBusID, pr.pciDeviceID);
  say("FN71 verify: file %s (%.2f GiB), card %s %s\n", argv[1], fsz / 1073741824.0, bdf, pr.gcnArchName);
  g_want_bdf = ((uint32_t)pr.pciBusID << 8) | ((uint32_t)pr.pciDeviceID << 3);
  int rc = 0;
  if (sha256("abc", 3) != "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") {
    say("FAIL: sha256 self-test\n");
    return 3;
  }

  // ---------------------------------------------------------------- A: the pool
  std::string sp = std::string("/sys/bus/pci/devices/") + bdf + "/p2pmem/";
  say("\n[A] sysfs %s: size %s available %s published %s\n", sp.c_str(), slurp(sp + "size").c_str(),
      slurp(sp + "available").c_str(), slurp(sp + "published").c_str());
  size_t pool_sz = strtoull(slurp(sp + "size").c_str(), nullptr, 0);
  std::string rn = render_node_for(bdf);
  int rfd = rn.empty() ? -1 : open(rn.c_str(), O_RDWR | O_CLOEXEC);
  say("    render node %s fd %d\n", rn.c_str(), rfd);
  if (rfd < 0 || pool_sz == 0) {
    say("FAIL: no render node or no pool (is amdgpu.p2pmem_mb set on this boot?)\n");
    return 4;
  }
  GemCreate gc;
  memset(&gc, 0, sizeof gc);
  gc.in.bo_size = pool_sz + 4096;
  gc.in.domains = DOMAIN_VRAM;
  gc.in.domain_flags = P2PMEM_LOCAL;
  int wrong = ioctl(rfd, FN71_GEM_CREATE, &gc);
  int wrong_e = wrong ? errno : 0;
  say("    GEM_CREATE(P2PMEM_LOCAL, wrong size): %d %s (want EINVAL)\n", wrong, wrong ? strerror(wrong_e) : "");
  if (wrong_e != EINVAL) rc = 5;
  memset(&gc, 0, sizeof gc);
  gc.in.bo_size = 0;
  gc.in.domains = DOMAIN_VRAM;
  gc.in.domain_flags = P2PMEM_LOCAL;
  if (ioctl(rfd, FN71_GEM_CREATE, &gc)) {
    say("FAIL: GEM_CREATE(P2PMEM_LOCAL): %s\n", strerror(errno));
    return 4;
  }
  PrimeHandle ph = {gc.out.handle, O_CLOEXEC | O_RDWR, -1};
  if (ioctl(rfd, FN71_PRIME_HANDLE_TO_FD, &ph)) {
    say("FAIL: PRIME_HANDLE_TO_FD: %s\n", strerror(errno));
    return 4;
  }
  say("    GEM handle %u -> dma-buf fd %d\n", gc.out.handle, ph.fd);
  // HIP import (takes its own fd)
  void *dev = nullptr;
  {
    hipExternalMemoryHandleDesc d;
    memset(&d, 0, sizeof d);
    d.type = hipExternalMemoryHandleTypeOpaqueFd;
    d.handle.fd = dup(ph.fd);
    d.size = pool_sz;
    hipExternalMemory_t em;
    hipError_t e = hipImportExternalMemory(&em, &d);
    if (e == hipSuccess) {
      hipExternalMemoryBufferDesc bd;
      memset(&bd, 0, sizeof bd);
      bd.offset = 0;
      bd.size = pool_sz;
      e = hipExternalMemoryGetMappedBuffer(&dev, em, &bd);
    }
    say("    hipImportExternalMemory + GetMappedBuffer: %s ptr %p\n", hipGetErrorString(e), dev);
  }
  // HSA interop (saber-hsa's path)
  void *hdev = nullptr;
  if (hsa_init() == HSA_STATUS_SUCCESS) {
    hsa_iterate_agents(agent_cb, nullptr);
    size_t hsz = 0, msz = 0;
    const void *meta = nullptr;
    hsa_status_t s = g_gpu.handle ? hsa_amd_interop_map_buffer(1, &g_gpu, dup(ph.fd), 0, &hsz, &hdev, &msz, &meta)
                                  : HSA_STATUS_ERROR_INVALID_AGENT;
    say("    hsa_amd_interop_map_buffer: status %d size %zu ptr %p\n", (int)s, hsz, hdev);
    if (s != HSA_STATUS_SUCCESS) hdev = nullptr;
  }
  if (!dev) dev = hdev;
  if (!dev) {
    say("FAIL: no GPU pointer to the pool\n");
    return 4;
  }
  int pfd = open(argv[3], O_RDWR | O_CLOEXEC);
  char *pool = pfd < 0 ? (char *)MAP_FAILED
                       : (char *)mmap(nullptr, pool_sz, PROT_READ | PROT_WRITE, MAP_SHARED, pfd, 0);
  say("    pool file %s fd %d, mmap(whole pool %zu B): %s\n", argv[3], pfd, pool_sz,
      pool == MAP_FAILED ? strerror(errno) : "ok");
  if (pool == MAP_FAILED) return 4;
  say("    sysfs available after the mmap: %s (want 0)\n", slurp(sp + "available").c_str());

  void *scratch = nullptr;
  const size_t W = std::min<size_t>(1024 * MiB, pool_sz);
  CK(hipMalloc(&scratch, W));
  char *host = (char *)aligned_alloc(4096, W);
  char *truth = (char *)aligned_alloc(4096, W);

  // ---------------------------------------------------------------- B: aliasing
  {
    size_t nw = pool_sz / 4;
    k_fill<<<1024, 256>>>((uint32_t *)dev, nw, 0x5a17u);
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());
    size_t bad = 0, seen = 0;
    for (size_t o = 0; o + 256 <= pool_sz; o += MiB) {
      volatile uint32_t *p = (volatile uint32_t *)(pool + o);
      for (size_t j = 0; j < 64; ++j, ++seen)
        if (p[j] != pat(o / 4 + j, 0x5a17u)) bad++;
    }
    say("\n[B] GPU wrote, CPU read through the pool mapping: %zu/%zu words differ -> %s\n", bad, seen,
        bad ? "NOT ALIASED" : "ALIASED (offset x == x)");
    if (bad) rc = 6;
    const size_t X = 4 * MiB, at = pool_sz / 2 / 4096 * 4096;
    uint32_t *c = (uint32_t *)(pool + at);
    for (size_t j = 0; j < X / 4; ++j) c[j] = pat(j, 0x77u);
    (void)((volatile uint32_t *)c)[X / 4 - 1];
    gpu_readback((char *)dev + at, host, X, scratch);
    size_t bad2 = 0;
    for (size_t j = 0; j < X / 4; ++j) bad2 += ((uint32_t *)host)[j] != pat(j, 0x77u);
    say("    CPU wrote 4 MiB at +%zu, the GPU (HIP ptr) read it: %zu words differ\n", at, bad2);
    if (bad2) rc = 6;
    if (hdev && hdev != dev) {
      gpu_readback((char *)hdev + at, host, X, scratch);
      size_t bad3 = 0;
      for (size_t j = 0; j < X / 4; ++j) bad3 += ((uint32_t *)host)[j] != pat(j, 0x77u);
      say("    ... the GPU through the HSA interop ptr: %zu words differ\n", bad3);
      if (bad3) rc = 6;
    }
  }

  // ---------------------------------------------------------------- C: the direct read
  ssize_t d1 = pread(g_fd_direct, pool, REC, 0);
  int e1 = d1 < 0 ? errno : 0;
  say("\n[C] pread(O_DIRECT, %zu B) into the pool: %zd %s\n", REC, d1, e1 ? strerror(e1) : "");
  if (d1 != (ssize_t)REC) {
    say("FAIL: the drive could not write the pool (P2P refused)\n");
    return 7;
  }

  // ---------------------------------------------------------------- D: correctness
  {
    double a = now_s();
    bool ok = read_span(pool, 0, W);
    double dt = now_s() - a;
    gpu_readback(dev, host, W, scratch);
    bool ok2 = read_span(truth, 0, W);
    std::string hb = sha256(host, W), ht = sha256(truth, W);
    bool eq = ok && ok2 && memcmp(host, truth, W) == 0;
    say("\n[D] %zu MiB of the file at 0 -> pool (QD8 8 MiB, %.2f GB/s, reads %s); GPU readback sha256 %s\n"
        "    file bytes (drive -> host RAM)  sha256 %s -> %s\n",
        W / MiB, W / dt / 1e9, ok ? "ok" : "SHORT", hb.c_str(), ht.c_str(), eq ? "BYTE-EQUAL" : "DIFFERENT");
    if (!eq) rc = 8;
  }

  // ---------------------------------------------------------------- E: the rewrite law (FN71b form)
  uint32_t *flag = nullptr, *ack = nullptr;
  CK(hipHostMalloc((void **)&flag, 4096, hipHostMallocCoherent | hipHostMallocMapped));
  CK(hipHostMalloc((void **)&ack, 4096, hipHostMallocCoherent | hipHostMallocMapped));
  {
    // E1: a fresh kernel after the overwrite, the lines WARM in GL2 (read twice just before).
    const size_t X = 2 * MiB;
    char *alt = (char *)aligned_alloc(4096, X);
    say("\n[E1] rewrite, fresh kernel: the GPU reads 2 MiB of the pool twice (GL2 warm), the drive overwrites it\n"
        "     with other file bytes, a NEW kernel reads it\n");
    for (int rep = 0; rep < 6; ++rep) {
      size_t po = (size_t)rep * 16 * MiB + (rep & 1) * 4096;
      uint64_t fa = (uint64_t)rep * 64 * MiB, fb = fa + 1024 * MiB + (uint64_t)(rep + 1) * 4096 * 7;
      read_span(pool + po, fa, X);
      gpu_copy((char *)dev + po, scratch, X);
      gpu_copy((char *)dev + po, scratch, X);
      read_span(alt, fa, X);
      read_span(truth, fb, X);
      size_t moved = 0;
      for (size_t j = 0; j < X; ++j) moved += alt[j] != truth[j];
      bool ok = read_span(pool + po, fb, X);
      gpu_readback((char *)dev + po, host, X, scratch);
      size_t diff = 0;
      for (size_t j = 0; j < X; ++j) diff += host[j] != truth[j];
      say("     rep %d pool +%zu: overwrite changed %zu bytes; the new kernel sees %zu stale bytes -> %s%s\n", rep, po,
          moved, diff, diff ? "STALE" : "EQUAL", ok ? "" : " (short read)");
      if (diff) rc = rc ? rc : 9;
    }
    // E2: the SAME kernel waits on the completion word, then re-reads (a consumer that reuses a window
    // without a launch boundary).
    uint4 *o1 = nullptr, *o2 = nullptr;
    CK(hipMalloc((void **)&o1, X));
    CK(hipMalloc((void **)&o2, X));
    hipStream_t st;
    CK(hipStreamCreateWithFlags(&st, hipStreamNonBlocking));
    const char *mname[3] = {"volatile spin, no fence", "spin + __threadfence_system()",
                            "acquire-system load in every thread"};
    say("[E2] rewrite inside ONE kernel (warm read -> ack -> host pread overwrites -> completion word -> re-read)\n");
    for (size_t sz : {(size_t)16384, X})
      for (int mode = 0; mode < 3; ++mode)
        for (int rep = 0; rep < 3; ++rep) {
          size_t po = 256 * MiB + (size_t)(mode * 3 + rep) * 4 * MiB;
          uint64_t fa = (uint64_t)(mode * 3 + rep + 1) * 96 * MiB, fb = fa + 2048 * MiB;
          read_span(pool + po, fa, sz);
          read_span(truth, fb, sz);
          *(volatile uint32_t *)flag = 0;
          *(volatile uint32_t *)ack = 0;
          k_spin_reread<<<1, 256, 0, st>>>((const uint4 *)((char *)dev + po), o1, o2, sz / 16, flag, ack, mode);
          CK(hipGetLastError());
          double t0 = now_s();
          bool to = false;
          while (*(volatile uint32_t *)ack == 0)
            if (now_s() - t0 > 10) {
              to = true;
              break;
            }
          bool ok = read_span(pool + po, fb, sz);
          std::atomic_thread_fence(std::memory_order_seq_cst);
          *(volatile uint32_t *)flag = 1;
          CK(hipStreamSynchronize(st));
          CK(hipMemcpy(host, o2, sz, hipMemcpyDeviceToHost));
          size_t diff = 0;
          for (size_t j = 0; j < sz; ++j) diff += host[j] != truth[j];
          say("     %7zu B  mode %d (%-36s) rep %d: %zu stale bytes -> %s%s%s\n", sz, mode, mname[mode], rep, diff,
              diff ? "STALE" : "EQUAL", ok ? "" : " (short read)", to ? " (ack TIMEOUT)" : "");
        }
    CK(hipFree(o1));
    CK(hipFree(o2));
    free(alt);

    // H: single-record latency to the GPU's first byte (QD1, 200 records): p2p + a resident kernel
    // waiting on the completion word; p2p + a kernel launched after the pread; the bounce
    // (pread -> pinned, H2D, kernel). Time = before the pread .. the host sees the kernel's ack.
    uint4 *ob = nullptr;
    CK(hipMalloc((void **)&ob, 64));
    void *pin = nullptr;
    CK(hipHostMalloc(&pin, REC, hipHostMallocDefault));
    char *vrec = nullptr;
    CK(hipMalloc((void **)&vrec, REC));
    uint64_t hspan = std::min<uint64_t>((uint64_t)fsz, 8ull << 30);
    say("\n[H] one %zu B record, QD1, 200 records: time from the pread's start to the GPU's ack of the first byte\n",
        REC);
    const char *fn[3] = {"p2p + resident kernel on the completion word", "p2p + kernel launched after the pread",
                         "bounce (pread->pinned, H2D, kernel)"};
    for (int form = 0; form < 3; ++form) {
      Lat l;
      for (int k = 0; k < 200; ++k) {
        uint64_t off = (((uint64_t)k * 0x9E3779B97F4A7C15ull) % (hspan / REC)) * REC;
        size_t po = 512 * MiB + (size_t)(k % 64) * REC;
        *(volatile uint32_t *)flag = 0;
        *(volatile uint32_t *)ack = 0;
        if (form == 0) {
          k_first_byte<<<1, 64, 0, st>>>((const uint4 *)((char *)dev + po), REC / 16, flag, ack, ob);
          CK(hipGetLastError());
          std::this_thread::sleep_for(std::chrono::microseconds(300));
        }
        double a = now_s();
        char *dst = form == 2 ? (char *)pin : pool + po;
        ssize_t r = pread(g_fd_direct, dst, REC, off);
        if (r != (ssize_t)REC) say("     short read %zd\n", r);
        if (form == 0) {
          std::atomic_thread_fence(std::memory_order_seq_cst);
          *(volatile uint32_t *)flag = 1;
        } else if (form == 1) {
          k_first_byte<<<1, 64, 0, st>>>((const uint4 *)((char *)dev + po), REC / 16, nullptr, ack, ob);
        } else {
          CK(hipMemcpyAsync(vrec, pin, REC, hipMemcpyHostToDevice, st));
          k_first_byte<<<1, 64, 0, st>>>((const uint4 *)vrec, REC / 16, nullptr, ack, ob);
        }
        while (*(volatile uint32_t *)ack == 0) {
        }
        l.v.push_back((now_s() - a) * 1e6);
        CK(hipStreamSynchronize(st));
      }
      say("     %-48s p50 %6.0f us  p99 %6.0f us\n", fn[form], l.pct(0.5), l.pct(0.99));
    }
    CK(hipFree(ob));
    CK(hipFree(vrec));
    CK(hipHostFree(pin));
    CK(hipStreamDestroy(st));
  }

  // ---------------------------------------------------------------- F: rates
  char *vram = nullptr;
  CK(hipMalloc((void **)&vram, 256 * MiB));
  uint64_t span = std::min<uint64_t>((uint64_t)fsz, 8ull << 30);
  say("\n[F] records: %zu B at pseudo-random 4 KiB-aligned offsets over %.1f GiB; 1024 records a point\n", REC,
      span / 1073741824.0);
  std::vector<Form> forms = {DRIVE_ONLY, BOUNCE, P2P};
  for (Form f : forms)
    for (int qd : {1, 8, 32}) {
      Lat l;
      double gbs = run_form(f, qd, REC, 1024, span, vram, pool, pool_sz, &l);
      say("  %-28s QD%-2d %6.2f GB/s  per-record p50 %7.0f us  p99 %7.0f us\n", form_name(f), qd, gbs, l.pct(0.5),
          l.pct(0.99));
    }
  say("  1 GiB stream: 128 x 8 MiB, QD 8\n");
  for (Form f : forms) {
    double gbs = run_form(f, 8, 8 * MiB, 128, span, vram, pool, pool_sz, nullptr);
    say("  %-28s %6.2f GB/s\n", form_name(f), gbs);
  }

  // ---------------------------------------------------------------- G: host DRAM contention
  {
    const size_t DB = 2048 * MiB;
    uint64_t *buf = (uint64_t *)aligned_alloc(4096, DB);
    memset(buf, 1, DB);
    auto dram_rate = [&](auto &&beside) {
      g_dram_bytes = 0;
      g_dram_run = true;
      std::vector<std::thread> th;
      const int T = 8;
      for (int t = 0; t < T; ++t) th.emplace_back(dram_thread, buf + t * (DB / 8 / T), DB / 8 / T);
      std::this_thread::sleep_for(std::chrono::milliseconds(300));
      uint64_t b0 = g_dram_bytes.load();
      double a = now_s();
      double other = beside();
      double dt = now_s() - a;
      uint64_t b1 = g_dram_bytes.load();
      g_dram_run = false;
      for (auto &x : th) x.join();
      return std::make_pair((b1 - b0) / dt / 1e9, other);
    };
    say("\n[G] host DRAM contention: 8 CPU threads stream-read 2 GiB; GB/s alone vs beside a transfer (~3 s)\n");
    auto alone = dram_rate([]() {
      std::this_thread::sleep_for(std::chrono::seconds(3));
      return 0.0;
    });
    say("  DRAM stream alone                      %6.2f GB/s\n", alone.first);
    for (Form f : forms) {
      auto r = dram_rate([&]() { return run_form(f, 32, REC, 1024, span, vram, pool, pool_sz, nullptr, 3.0); });
      say("  DRAM stream beside %-28s QD32 %6.2f GB/s (transfer %6.2f GB/s)\n", form_name(f), r.first, r.second);
    }
    free(buf);
  }
  munmap(pool, pool_sz);
  say("\n%s (rc %d)\n", rc ? "DONE WITH FAILURES" : "DONE: all checks pass", rc);
  return rc;
}
