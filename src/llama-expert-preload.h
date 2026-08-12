#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <string>

struct ggml_tensor;
struct ggml_backend_buffer;
struct ggml_backend_buffer_type;

// pre-load handoff between the model loader and the expert hot store.
//
// with the tier active (--expert-hot-s set), the expert weight tensors stay
// fully CPU-resident (the -cmoe override keeps them on a host buffer, GPU
// overrides are filtered by the hot store). the loader registers each tensor
// and precomputes a ground-truth hash of every expert slice; the cold op and
// the hot store read the slices straight from the tensor payloads, and move
// mode (--expert-move-mode 2) releases the hot rows' RAM pages once they live
// on the GPU (re-read from the gguf file on demand). there is no separate
// streaming store anymore: the tensors ARE the cold store.
namespace llama_expert_preload {

    LLAMA_API void set_slots(int s);
    int  get_slots();
    LLAMA_API void set_model_path(const char * path); // for the debug disk hash
    LLAMA_API void set_no_evict(bool no_evict);       // --expert-no-evict
    bool get_no_evict();

    // min compute capability floor (sm_70) for the GPU hot store: below it the
    // store must not engage (Pascal sm_61 regresses; see RFC #25857). cc is
    // 100*major+10*minor for NVIDIA CUDA (610, 750, ...); AMD/MUSA offsets are
    // far above this, so those always pass.
    static constexpr int EXPERT_MIN_CC = 700;

    // minimum cc (100*major+10*minor) among the CUDA GPU devices that would
    // host the hot store, honoring expert_gpu pinning (same filter as the
    // hotstore's gpu_bufts build in llama-context.cpp). returns 0 when no CUDA
    // device qualifies, i.e. the gate does not apply.
    LLAMA_API int gpu_min_cc(int expert_gpu);

    // parse the --expert-gpu value: "-1"/"all" -> -1 (all GPUs), a plain
    // integer -> itself, or a backend device name like CUDA0 -> its index among
    // the GPU devices (same ordering as gpu_min_cc). throws std::invalid_argument
    // on an unknown device or a non-GPU device.
    LLAMA_API int expert_gpu_parse(const std::string & sel);

    // true if `name` matches an exps weight tensor; sets layer_idx on match
    bool is_exps(const char * name, int & layer_idx);

    struct entry {
        const ggml_tensor * src;
        size_t plane_bytes;  // bytes of one expert slice in this tensor
        size_t file_off;     // absolute gguf data offset of this tensor
        int    fd;           // model file descriptor (for the debug disk read)
        int    n_experts;
    };

    // loader side: register a CPU-resident exps tensor and precompute the
    // ground-truth hash of every expert slice from `data` (the tensor payload).
    // returns the entry index, or -1 when `data` is null (GPU-resident tensor:
    // not host-backed, the hot store filters it the same way).
    int    register_tensor(const ggml_tensor * src, size_t plane_bytes, int n_experts,
                           size_t file_off, int fd, const uint8_t * data);
    bool   read_expert(size_t idx, int expert, void * out, size_t n); // from the gguf file

    // hotstore + cold-op side.
    int    index_of(const ggml_tensor * src);

    // cold expert slice access (nullptr when the slice is on the GPU)
    const uint8_t * cpu_slice(size_t idx, int expert);
    void free_cpu_slice(size_t idx, int expert);
    void set_cpu_slice(size_t idx, int expert, const uint8_t * data);

    // debug: ground-truth FNV-1a of the first 1024 bytes of an expert slice,
    // taken straight from the GGUF at load. compare against the hash of the
    // slice actually read to verify a copy/routing is correct.
    uint64_t expected_hash(const ggml_tensor * src, int expert);

    void clear();

} // namespace llama_expert_preload
