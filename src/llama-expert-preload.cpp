#include "llama-expert-preload.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstring>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include <sys/stat.h>
// MSVC POSIX layer: map open/close to _open/_close and emulate pread
#define open  _open
#define close _close
#define O_RDONLY _O_RDONLY
static long long pread(int fd, void * buf, size_t n, long long off) {
    if (_lseeki64(fd, off, SEEK_SET) < 0) {
        return -1;
    }
    return _read(fd, buf, (unsigned int) n);
}
#else
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#endif

namespace llama_expert_preload {

    int index_of(const ggml_tensor * src);

namespace {
    // matches an expert weight tensor, e.g. blk.0.ffn_gate_exps.weight
    const std::regex g_re_exps("blk\\.(\\d+)\\.ffn_(up|down|gate|gate_up)_(ch|)exps\\.weight");

    int g_slots = 0;

    std::vector<entry> g_entries;
    std::unordered_map<const ggml_tensor *, size_t> g_src_idx; // src -> entry index
    std::vector<std::vector<uint64_t>> g_hashes; // [entry][expert] -> gguf FNV-1a(1024B)

    static uint64_t fnv1a(const uint8_t * p, size_t n) {
        uint64_t h = 0xcbf29ce484222325ULL;
        for (size_t i = 0; i < n; i++) {
            h ^= p[i];
            h *= 0x100000001b3ULL;
        }
        return h;
    }

    std::string g_path;
    int g_fd = -1;

    // C callback for the ggml cold op: resolve a cold expert's address from
    // the tensor payload (the tensors ARE the cold store)
    const uint8_t * preload_slice_cb(const struct ggml_tensor * src0, int expert) {
        if (!src0 || expert < 0) {
            return nullptr;
        }
        const int i = index_of(src0);
        if (i < 0) {
            return nullptr;
        }
        return cpu_slice((size_t) i, expert);
    }
}

LLAMA_API void set_model_path(const char * path) {
    g_path = path ? path : "";
    if (g_fd >= 0) {
        close(g_fd);
        g_fd = -1;
    }
    g_fd = open(g_path.c_str(), O_RDONLY);
}
LLAMA_API void set_slots(int s) {
    g_slots = s;
}

int get_slots() {
    return g_slots;
}

bool g_no_evict = false;

LLAMA_API void set_no_evict(bool no_evict) {
    g_no_evict = no_evict;
}

bool get_no_evict() {
    return g_no_evict;
}

int gpu_min_cc(int expert_gpu) {
    int min_cc = 0;
    int gpu_idx = 0; // index among GPU (non-CPU/non-ACCEL) devices, like the hotstore
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        const ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) {
            continue;
        }
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if (type == GGML_BACKEND_DEVICE_TYPE_CPU || type == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            continue;
        }
        if (expert_gpu >= 0 && gpu_idx != expert_gpu) {
            gpu_idx++;
            continue; // store pinned to a specific GPU: skip the others
        }
        gpu_idx++;

        // only the CUDA backend exposes cc; other GPU backends add no constraint
        const ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        if (!reg) {
            continue;
        }
        typedef int (* get_cc_fn)(int);
        const get_cc_fn get_cc = (get_cc_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_get_device_cc");
        if (!get_cc) {
            continue;
        }
        // find this device's index within its backend reg
        for (size_t j = 0; j < ggml_backend_reg_dev_count(reg); ++j) {
            if (ggml_backend_reg_dev_get(reg, j) == dev) {
                const int cc = get_cc((int) j);
                if (cc > 0 && (min_cc == 0 || cc < min_cc)) {
                    min_cc = cc;
                }
                break;
            }
        }
    }
    return min_cc;
}

int expert_gpu_parse(const std::string & sel) {
    if (sel.empty() || sel == "-1" || sel == "all") {
        return -1;
    }
    // plain integer, optionally negative (any negative = all GPUs, like -1)
    const bool digits_only = std::all_of(sel.begin(), sel.end(), [](char c) { return c >= '0' && c <= '9'; });
    const bool neg_int = sel.size() > 1 && sel[0] == '-' &&
        std::all_of(sel.begin() + 1, sel.end(), [](char c) { return c >= '0' && c <= '9'; });
    if (digits_only || neg_int) {
        const int v = std::stoi(sel);
        return v < 0 ? -1 : v;
    }

    // backend device name like CUDA0, CUDA1, Vulkan0, ...
    ggml_backend_load_all();
    const ggml_backend_dev_t dev = ggml_backend_dev_by_name(sel.c_str());
    if (!dev) {
        throw std::invalid_argument("invalid --expert-gpu device: " + sel);
    }
    int gpu_idx = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        const ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (!d) {
            continue;
        }
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(d);
        if (type == GGML_BACKEND_DEVICE_TYPE_CPU || type == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            continue;
        }
        if (d == dev) {
            return gpu_idx;
        }
        gpu_idx++;
    }
    throw std::invalid_argument("device is not a GPU: " + sel);
}

bool is_exps(const char * name, int & layer_idx) {
    if (!name) {
        return false;
    }
    std::cmatch m;
    if (!std::regex_search(name, m, g_re_exps)) {
        return false;
    }
    layer_idx = std::stoi(m[1].str());
    return true;
}

int register_tensor(const ggml_tensor * src, size_t plane_bytes, int n_experts,
                    size_t file_off, int fd, const uint8_t * data, bool file_backed) {
    if (!src || !data || plane_bytes == 0 || n_experts <= 0) {
        return -1; // GPU-resident or degenerate tensor: not host-backed
    }
    for (size_t i = 0; i < g_entries.size(); i++) {
        if (g_entries[i].src == src) {
            g_entries[i].fd = fd; // refresh: the file may be reopened between passes
            return (int) i; // already registered (second load pass)
        }
    }
    g_entries.push_back({src, plane_bytes, file_off, fd, n_experts, file_backed});
    g_src_idx[src] = g_entries.size() - 1;
    g_hashes.emplace_back((size_t) n_experts, 0);
    const size_t chunk = plane_bytes < 1024 ? plane_bytes : 1024;
    for (int ex = 0; ex < n_experts; ex++) {
        g_hashes.back()[ex] = fnv1a(data + (size_t) ex * plane_bytes, chunk);
    }
    ggml_mmid_cold_set_slice_fn(preload_slice_cb);
    return (int) g_entries.size() - 1;
}

bool read_expert(size_t idx, int expert, void * out, size_t n) {
    if (idx >= g_entries.size() || expert < 0 || expert >= g_entries[idx].n_experts) {
        return false;
    }
    const entry & e = g_entries[idx];
    if (n > e.plane_bytes) {
        return false;
    }
    const long long got = pread(g_fd >= 0 ? g_fd : e.fd, out, n,
        (long long) (e.file_off + (size_t) expert * e.plane_bytes));
    return got == (long long) n;
}

int index_of(const ggml_tensor * src) {
    if (!src) {
        return -1;
    }
    for (size_t i = 0; i < g_entries.size(); i++) {
        if (g_entries[i].src == src) {
            return (int) i;
        }
    }
    // the hotstore's entry tensors can be distinct objects with the same name
    // (created in a different context); fall back to matching by name
    for (size_t i = 0; i < g_entries.size(); i++) {
        if (g_entries[i].src && g_entries[i].src->name[0] &&
            strcmp(g_entries[i].src->name, src->name) == 0) {
            return (int) i;
        }
    }
    return -1;
}

bool is_file_backed(size_t idx) {
    if (idx >= g_entries.size()) {
        return false;
    }
    return g_entries[idx].file_backed;
}

uint64_t expected_hash(const ggml_tensor * src, int expert) {
    const int idx = index_of(src);
    if (idx < 0 || expert < 0 || expert >= (int) g_hashes[idx].size()) {
        return 0;
    }
    return g_hashes[idx][expert];
}

const uint8_t * cpu_slice(size_t idx, int expert) {
    if (idx >= g_entries.size() || !g_entries[idx].src) {
        return nullptr;
    }
    const entry & e = g_entries[idx];
    if (expert < 0 || expert >= e.n_experts) {
        return nullptr;
    }
    const uint8_t * data = (const uint8_t *) ggml_get_data(e.src);
    if (!data) {
        return nullptr;
    }
    return data + (size_t) expert * e.plane_bytes;
}

static void release_pages(void * ptr, size_t len) {
#ifdef _WIN32
    // Windows maps the gguf tensor payloads into a file-backed read-only view;
    // DiscardVirtualMemory (Win8+) drops those cache pages so their RAM is
    // reclaimed, and re-reads them from the gguf file on the next access. this
    // is the semantic equivalent of madvise(MADV_DONTNEED) and, unlike
    // VirtualFree(MEM_RESET), is valid on file-backed sections.
    if (!ptr || len == 0) {
        return;
    }
    typedef BOOL (WINAPI * pfn_discard)(const void *, SIZE_T);
    static pfn_discard pDiscardVirtualMemory = (pfn_discard) GetProcAddress(GetModuleHandleA("kernel32.dll"), "DiscardVirtualMemory");
    if (pDiscardVirtualMemory) {
        pDiscardVirtualMemory(ptr, len); // best-effort; failure leaves the pages resident
    }
#else
    const long page = sysconf(_SC_PAGESIZE);
    const uintptr_t base   = (uintptr_t) ptr;
    const uintptr_t start  = (base + (uintptr_t) page - 1) & ~((uintptr_t) page - 1);
    const uintptr_t end    = (base + len) & ~((uintptr_t) page - 1);
    if (start < end) {
        madvise((void *) start, end - start, MADV_DONTNEED);
    }
#endif
}

void free_cpu_slice(size_t idx, int expert) {
    if (idx >= g_entries.size() || !g_entries[idx].src) {
        return;
    }
    const entry & e = g_entries[idx];
    if (expert < 0 || expert >= e.n_experts) {
        return;
    }
    uint8_t * data = (uint8_t *) ggml_get_data(e.src);
    if (!data) {
        return;
    }
    release_pages(data + (size_t) expert * e.plane_bytes, e.plane_bytes);
}

void set_cpu_slice(size_t idx, int expert, const uint8_t * data) {
#ifdef _WIN32
    // Windows maps the gguf tensor payloads into a read-only file mapping and
    // never relinquishes the RAM copy (release_pages is a no-op below), so the
    // expert bytes are always resident and this write-back is both redundant
    // and illegal: memcpy into a PAGE_READONLY view faults (access violation).
    (void) idx;
    (void) expert;
    (void) data;
    return;
#endif
    if (idx >= g_entries.size() || !g_entries[idx].src || !data) {
        return;
    }
    const entry & e = g_entries[idx];
    if (expert < 0 || expert >= e.n_experts) {
        return;
    }
    uint8_t * dst = (uint8_t *) ggml_get_data(e.src);
    if (!dst) {
        return;
    }
    std::memcpy(dst + (size_t) expert * e.plane_bytes, data, e.plane_bytes);
}

void clear() {
    g_entries.clear();
    g_src_idx.clear();
    g_hashes.clear();
}

} // namespace llama_expert_preload