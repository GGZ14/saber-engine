// Bit-exact gate: replay Saber's own code objects against the prod kernel's captured output.
//
// Reads a JSON array of LaunchDescriptors produced by `saber-kplan` (i.e. by the Rust `resolve`
// that the engine will call), loads the named `.hsaco` with hipModuleLoad, uploads the byte
// buffers the internal reference harness dumped from the REFERENCE launcher, launches exactly what
// the descriptor says -- kernel symbol, grid, block, kernarg byte layout -- and memcmp's the
// output against the reference bytes.
//
// It re-derives NOTHING: no tile choice, no split-K rule, no kernarg offset. If the descriptor is
// wrong the gate fails, which is the point.
//
// Build (inside the dev image):
//   hipcc -O2 -std=c++17 crates/saber-kernels/hip/harness/run_hsaco.cpp -o .cache/capture/run_hsaco
// Run:
//   HIP_VISIBLE_DEVICES=1 run_hsaco --plan plan.json --data .cache/capture --prebuilt prebuilt/gfx1201
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// ------------------------------------------------------------------ a very small JSON reader
// Enough for saber-kplan's output: arrays, objects, strings, numbers, bools. No unicode escapes
// beyond the ones a kernel name could contain (none), and every miss is fatal.
struct Json {
  enum Kind { Null, Bool, Num, Str, Arr, Obj } kind = Null;
  bool b = false;
  double num = 0;
  std::string str;
  std::vector<Json> arr;
  std::vector<std::pair<std::string, Json>> obj;

  const Json *get(const std::string &k) const {
    for (auto &kv : obj)
      if (kv.first == k) return &kv.second;
    return nullptr;
  }
  const Json &at(const std::string &k) const {
    const Json *v = get(k);
    if (!v) {
      fprintf(stderr, "run_hsaco: descriptor has no `%s`\n", k.c_str());
      exit(2);
    }
    return *v;
  }
  long i(const std::string &k) const { return (long)at(k).num; }
  const std::string &s(const std::string &k) const { return at(k).str; }
};

struct JsonParser {
  const std::string &t;
  size_t p = 0;
  explicit JsonParser(const std::string &s) : t(s) {}
  void ws() { while (p < t.size() && (unsigned char)t[p] <= ' ') ++p; }
  [[noreturn]] void die(const char *m) {
    fprintf(stderr, "run_hsaco: JSON at %zu: %s\n", p, m);
    exit(2);
  }
  Json value() {
    ws();
    if (p >= t.size()) die("eof");
    char c = t[p];
    if (c == '{') return object();
    if (c == '[') return array();
    if (c == '"') { Json j; j.kind = Json::Str; j.str = string(); return j; }
    if (c == 't' || c == 'f') {
      Json j; j.kind = Json::Bool; j.b = c == 't';
      p += (c == 't') ? 4 : 5;
      return j;
    }
    if (c == 'n') { p += 4; return Json(); }
    return number();
  }
  Json object() {
    Json j; j.kind = Json::Obj;
    ++p; ws();
    if (t[p] == '}') { ++p; return j; }
    for (;;) {
      ws();
      std::string k = string();
      ws();
      if (t[p] != ':') die("expected :");
      ++p;
      j.obj.emplace_back(k, value());
      ws();
      if (t[p] == ',') { ++p; continue; }
      if (t[p] == '}') { ++p; return j; }
      die("expected , or }");
    }
  }
  Json array() {
    Json j; j.kind = Json::Arr;
    ++p; ws();
    if (t[p] == ']') { ++p; return j; }
    for (;;) {
      j.arr.push_back(value());
      ws();
      if (t[p] == ',') { ++p; continue; }
      if (t[p] == ']') { ++p; return j; }
      die("expected , or ]");
    }
  }
  std::string string() {
    if (t[p] != '"') die("expected a string");
    ++p;
    std::string out;
    while (t[p] != '"') {
      if (t[p] == '\\') {
        ++p;
        char c = t[p++];
        switch (c) {
          case 'n': out.push_back('\n'); break;
          case 't': out.push_back('\t'); break;
          case 'u': p += 4; out.push_back('?'); break;
          default: out.push_back(c);
        }
      } else {
        out.push_back(t[p++]);
      }
    }
    ++p;
    return out;
  }
  Json number() {
    size_t s = p;
    while (p < t.size() && (isdigit((unsigned char)t[p]) || strchr("+-.eE", t[p]))) ++p;
    Json j; j.kind = Json::Num;
    j.num = strtod(t.substr(s, p - s).c_str(), nullptr);
    return j;
  }
};

// ------------------------------------------------------------------ flat uint8 .npy
static std::vector<uint8_t> read_npy_u8(const std::string &path, bool *ok) {
  std::ifstream f(path, std::ios::binary);
  *ok = false;
  if (!f) return {};
  char magic[6];
  f.read(magic, 6);
  if (memcmp(magic, "\x93NUMPY", 6) != 0) return {};
  uint8_t major = f.get(), minor = f.get();
  (void)minor;
  uint32_t hlen = 0;
  if (major == 1) {
    uint16_t h = 0;
    f.read((char *)&h, 2);
    hlen = h;
  } else {
    f.read((char *)&hlen, 4);
  }
  std::string header(hlen, '\0');
  f.read(&header[0], hlen);
  if (header.find("'|u1'") == std::string::npos && header.find("\"|u1\"") == std::string::npos) {
    fprintf(stderr, "run_hsaco: %s is not flat uint8 (%s)\n", path.c_str(), header.c_str());
    exit(2);
  }
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  *ok = true;
  return data;
}

#define HIP_OK(expr)                                                                       \
  do {                                                                                     \
    hipError_t _e = (expr);                                                                \
    if (_e != hipSuccess) {                                                                \
      fprintf(stderr, "run_hsaco: %s -> %s (%d) at %s:%d\n", #expr, hipGetErrorString(_e),  \
              (int)_e, __FILE__, __LINE__);                                                \
      exit(3);                                                                             \
    }                                                                                      \
  } while (0)

struct Buf {
  void *dev = nullptr;
  size_t bytes = 0;
};

int main(int argc, char **argv) {
  std::string plan_path, data_dir, prebuilt_dir, only;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&] { return std::string(argv[++i]); };
    if (a == "--plan") plan_path = next();
    else if (a == "--data") data_dir = next();
    else if (a == "--prebuilt") prebuilt_dir = next();
    else if (a == "--only") only = next();
    else { fprintf(stderr, "run_hsaco: unknown flag %s\n", a.c_str()); return 2; }
  }
  if (plan_path.empty() || data_dir.empty() || prebuilt_dir.empty()) {
    fprintf(stderr, "usage: run_hsaco --plan P --data DIR --prebuilt DIR [--only SUBSTR]\n");
    return 2;
  }
  std::ifstream pf(plan_path);
  std::stringstream ss;
  ss << pf.rdbuf();
  std::string text = ss.str();
  JsonParser jp(text);
  Json plan = jp.value();
  if (plan.kind != Json::Arr) { fprintf(stderr, "run_hsaco: plan is not an array\n"); return 2; }

  // Buffers a case shares with other cases (the weights are per shape, not per M).
  std::map<std::string, std::string> shared;
  {
    std::ifstream s(data_dir + "/shared.txt");
    std::string cid, name, file;
    while (s >> cid >> name >> file) shared[cid + "/" + name] = file;
  }

  std::map<std::string, hipModule_t> modules;
  int pass = 0, fail = 0, skipped = 0;
  printf("%-34s %-46s %10s %s\n", "case", "kernel", "bytes", "result");
  for (const Json &d : plan.arr) {
    const std::string task = d.s("task");
    std::string cid = task;
    for (char &c : cid) if (c == ' ') c = '_';
    if (!only.empty() && task.find(only) == std::string::npos) continue;
    if (d.get("error")) {
      printf("%-34s %-46s %10s REFUSED (%s)\n", task.c_str(), "-", "-", d.s("error").c_str());
      ++skipped;
      continue;
    }
    const std::string plugin = d.s("plugin"), kernel = d.s("kernel");

    // Inputs and expected outputs, by the buffer names the descriptor binds.
    std::map<std::string, Buf> bufs;
    std::vector<std::pair<std::string, std::vector<uint8_t>>> expects;
    bool missing = false;
    size_t total = 0;
    for (const Json &slot : d.at("args").arr) {
      if (slot.s("kind") != "buffer") continue;
      const std::string name = slot.s("buffer");
      if (bufs.count(name)) continue;
      bool ok = false;
      std::vector<uint8_t> in = read_npy_u8(data_dir + "/" + cid + ".in." + name + ".npy", &ok);
      if (!ok) {
        auto it = shared.find(cid + "/" + name);
        if (it != shared.end())
          in = read_npy_u8(data_dir + "/" + it->second + ".npy", &ok);
      }
      if (ok) {
        Buf b;
        b.bytes = in.size();
        HIP_OK(hipMalloc(&b.dev, b.bytes));
        HIP_OK(hipMemcpy(b.dev, in.data(), b.bytes, hipMemcpyHostToDevice));
        bufs[name] = b;
        total += b.bytes;
        // IN-PLACE buffers: the reference's GDN kernels rewrite the conv and recurrent state caches
        // where they found them, and the split-KV combine reads the partials the decode kernel
        // wrote. When BOTH an .in. and an .out. exist for one name, the .in. is uploaded and the
        // .out. is what the result must equal.
        bool ok2 = false;
        std::vector<uint8_t> after =
            read_npy_u8(data_dir + "/" + cid + ".out." + name + ".npy", &ok2);
        if (ok2) {
          if (after.size() != b.bytes) {
            printf("%-34s %-46s %10s SIZE MISMATCH on in-place `%s` (%zu in, %zu out)\n",
                   task.c_str(), kernel.c_str(), "-", name.c_str(), b.bytes, after.size());
            missing = true;
            break;
          }
          expects.emplace_back(name, std::move(after));
        }
        continue;
      }
      std::vector<uint8_t> want = read_npy_u8(data_dir + "/" + cid + ".out." + name + ".npy", &ok);
      if (ok) {
        Buf b;
        b.bytes = want.size();
        HIP_OK(hipMalloc(&b.dev, b.bytes));
        HIP_OK(hipMemset(b.dev, 0, b.bytes));
        bufs[name] = b;
        total += b.bytes;
        expects.emplace_back(name, std::move(want));
        continue;
      }
      // Not captured: it must be a workspace the descriptor declared, or the plan is wrong.
      bool found = false;
      if (const Json *ws = d.get("workspaces")) {
        for (const Json &w : ws->arr) {
          if (w.s("name") != name) continue;
          Buf b;
          b.bytes = (size_t)w.i("bytes");
          HIP_OK(hipMalloc(&b.dev, b.bytes));
          HIP_OK(hipMemset(b.dev, 0, b.bytes));   // Persistent: zeroed once, before any launch
          bufs[name] = b;
          total += b.bytes;
          found = true;
          break;
        }
      }
      if (!found) {
        printf("%-34s %-46s %10s NO DATA for buffer `%s`\n", task.c_str(), kernel.c_str(), "-",
               name.c_str());
        missing = true;
        break;
      }
    }
    if (missing) {
      for (auto &kv : bufs) hipFree(kv.second.dev);
      ++skipped;
      continue;
    }

    // Kernarg block, written straight from the descriptor's byte layout.
    std::vector<uint8_t> karg((size_t)d.i("kernarg_bytes"), 0);
    for (const Json &slot : d.at("args").arr) {
      const size_t off = (size_t)slot.i("offset");
      const size_t size = (size_t)slot.i("size");
      const std::string kind = slot.s("kind");
      if (off + size > karg.size()) { fprintf(stderr, "run_hsaco: arg past kernarg end\n"); return 2; }
      if (kind == "buffer") {
        void *p = bufs[slot.s("buffer")].dev;
        memcpy(karg.data() + off, &p, sizeof(void *));
      } else if (kind == "null_buffer") {
        // A pointer this instantiation provably never dereferences (DKS==1's partial slab and
        // block counter). Writing a null is the descriptor's claim; a kernel that did touch it
        // would fault here rather than corrupt someone else's allocation.
        void *p = nullptr;
        memcpy(karg.data() + off, &p, sizeof(void *));
      } else if (kind == "i32") {
        int32_t v = (int32_t)slot.at("ival").num;
        memcpy(karg.data() + off, &v, 4);
      } else if (kind == "i64") {
        int64_t v = (int64_t)slot.at("ival").num;
        memcpy(karg.data() + off, &v, 8);
      } else if (kind == "f32") {
        float v = (float)slot.at("fval").num;
        memcpy(karg.data() + off, &v, 4);
      } else {
        fprintf(stderr, "run_hsaco: unknown arg kind `%s`\n", kind.c_str());
        return 2;
      }
    }

    if (!modules.count(plugin)) {
      hipModule_t m;
      HIP_OK(hipModuleLoad(&m, (prebuilt_dir + "/" + plugin + ".hsaco").c_str()));
      modules[plugin] = m;
    }
    hipFunction_t fn;
    HIP_OK(hipModuleGetFunction(&fn, modules[plugin], kernel.c_str()));

    size_t karg_size = karg.size();
    void *extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, karg.data(), HIP_LAUNCH_PARAM_BUFFER_SIZE,
                     &karg_size, HIP_LAUNCH_PARAM_END};
    const Json &g = d.at("grid");
    const Json &b = d.at("block");
    HIP_OK(hipModuleLaunchKernel(fn, (unsigned)g.arr[0].num, (unsigned)g.arr[1].num,
                                 (unsigned)g.arr[2].num, (unsigned)b.arr[0].num,
                                 (unsigned)b.arr[1].num, (unsigned)b.arr[2].num,
                                 (unsigned)d.i("shared_bytes"), nullptr, nullptr, extra));
    HIP_OK(hipDeviceSynchronize());

    size_t bad = 0, first = 0, nbytes = 0;
    std::string worst;
    for (auto &e : expects) {
      std::vector<uint8_t> got(e.second.size());
      HIP_OK(hipMemcpy(got.data(), bufs[e.first].dev, got.size(), hipMemcpyDeviceToHost));
      nbytes += got.size();
      for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] != e.second[i]) {
          if (!bad) { first = i; worst = e.first; }
          ++bad;
        }
      }
    }
    if (bad == 0) {
      printf("%-34s %-46s %10zu OK\n", task.c_str(), kernel.c_str(), nbytes);
      ++pass;
    } else {
      printf("%-34s %-46s %10zu MISMATCH %zu bytes (first in `%s` at %zu)\n", task.c_str(),
             kernel.c_str(), nbytes, bad, worst.c_str(), first);
      ++fail;
    }
    for (auto &kv : bufs) HIP_OK(hipFree(kv.second.dev));
    (void)total;
  }
  printf("\n%d bit-exact, %d MISMATCH, %d skipped\n", pass, fail, skipped);
  return fail ? 1 : 0;
}
