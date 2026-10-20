/**
 * @file dsp_driver.cpp
 * @brief Dynamic loader and hardware implementation for Cadence Vision Q7 DSP execution.
 */

#include "dsp_driver.hpp"

#include <dlfcn.h>
#include <iostream>
#include <cstring>
#include <sstream>

namespace eswin {

namespace {

#define BUFFER_CNT_MAXSIZE 64
#define SIZE_2M 0x200000
#define ALIGN_2M(size) (((size) + (SIZE_2M - 1)) & ~(SIZE_2M - 1))

#pragma pack(push, 8)
struct RawDevBuf {
    uint64_t memFd;
    uint64_t offset;
    uint64_t size;
    uint64_t reserved;
};

struct RawDspTask {
    uint32_t operatorHandle;
    RawDevBuf dspBuffers[BUFFER_CNT_MAXSIZE];
    uint32_t bufferCntCfg;
    uint32_t bufferCntInput;
    uint32_t bufferCntOutput;
    uint32_t priority;
    uint32_t syncCache;
    uint32_t pollMode;
    uint32_t taskHandle;
    void* callback;
    void* cbArg;
    uint32_t reserved;
};
#pragma pack(pop)

using fn_ES_DSP_LL_GetVersion = int (*)(uint64_t* version);
using fn_ES_DSP_LL_GetVersionString = int (*)(char* str, uint32_t maxLen);
using fn_ES_DSP_LL_Open = int (*)(int dspId, int* dspFd);
using fn_ES_DSP_LL_Close = int (*)(int dspFd);
using fn_ES_DSP_LL_LoadOperator = int (*)(int dspFd, const char* opName, const char* opDir, uint32_t* handle);
using fn_ES_DSP_LL_UnloadOperator = int (*)(int dspFd, uint32_t handle);
using fn_ES_DSP_LL_PrepareDMABuffer = int (*)(int dspFd, RawDevBuf buf);
using fn_ES_DSP_LL_UnprepareDMABuffer = int (*)(int dspFd, uint64_t memFd);
using fn_ES_DSP_LL_SubmitTask = int (*)(int dspFd, RawDspTask* task);

using fn_ES_SYS_MemAlloc = int (*)(uint64_t* memFd, int cacheMode, const char* name, const char* zone, size_t size);
using fn_ES_SYS_MemFree = int (*)(uint64_t memFd);
using fn_ES_SYS_Mmap = void* (*)(uint64_t memFd, size_t size, int cacheMode);
using fn_ES_SYS_Munmap = int (*)(void* ptr, size_t size);

struct DspSymbols {
    fn_ES_DSP_LL_GetVersion GetVersion{nullptr};
    fn_ES_DSP_LL_GetVersionString GetVersionString{nullptr};
    fn_ES_DSP_LL_Open Open{nullptr};
    fn_ES_DSP_LL_Close Close{nullptr};
    fn_ES_DSP_LL_LoadOperator LoadOperator{nullptr};
    fn_ES_DSP_LL_UnloadOperator UnloadOperator{nullptr};
    fn_ES_DSP_LL_PrepareDMABuffer PrepareDMABuffer{nullptr};
    fn_ES_DSP_LL_UnprepareDMABuffer UnprepareDMABuffer{nullptr};
    fn_ES_DSP_LL_SubmitTask SubmitTask{nullptr};

    fn_ES_SYS_MemAlloc MemAlloc{nullptr};
    fn_ES_SYS_MemFree MemFree{nullptr};
    fn_ES_SYS_Mmap Mmap{nullptr};
    fn_ES_SYS_Munmap Munmap{nullptr};
};

static DspSymbols g_syms;
static bool g_syms_loaded = false;

void init_dsp_symbols() {
    if (g_syms_loaded) return;

    void* dsp_lib = dlopen("libes_dsp_runtime_ll.so", RTLD_NOW | RTLD_GLOBAL);
    if (!dsp_lib) {
        dsp_lib = dlopen("/usr/lib/libes_dsp_runtime_ll.so", RTLD_NOW | RTLD_GLOBAL);
    }
    if (!dsp_lib) {
        throw std::runtime_error(std::string("Failed to load libes_dsp_runtime_ll.so: ") + dlerror());
    }

    void* mem_lib = dlopen("libes_memory.so", RTLD_NOW | RTLD_GLOBAL);
    if (!mem_lib) {
        mem_lib = dlopen("/usr/lib/libes_memory.so", RTLD_NOW | RTLD_GLOBAL);
    }
    if (!mem_lib) {
        throw std::runtime_error(std::string("Failed to load libes_memory.so: ") + dlerror());
    }

    g_syms.GetVersion = reinterpret_cast<fn_ES_DSP_LL_GetVersion>(dlsym(dsp_lib, "ES_DSP_LL_GetVersion"));
    g_syms.GetVersionString = reinterpret_cast<fn_ES_DSP_LL_GetVersionString>(dlsym(dsp_lib, "ES_DSP_LL_GetVersionString"));
    g_syms.Open = reinterpret_cast<fn_ES_DSP_LL_Open>(dlsym(dsp_lib, "ES_DSP_LL_Open"));
    g_syms.Close = reinterpret_cast<fn_ES_DSP_LL_Close>(dlsym(dsp_lib, "ES_DSP_LL_Close"));
    g_syms.LoadOperator = reinterpret_cast<fn_ES_DSP_LL_LoadOperator>(dlsym(dsp_lib, "ES_DSP_LL_LoadOperator"));
    g_syms.UnloadOperator = reinterpret_cast<fn_ES_DSP_LL_UnloadOperator>(dlsym(dsp_lib, "ES_DSP_LL_UnloadOperator"));
    g_syms.PrepareDMABuffer = reinterpret_cast<fn_ES_DSP_LL_PrepareDMABuffer>(dlsym(dsp_lib, "ES_DSP_LL_PrepareDMABuffer"));
    g_syms.UnprepareDMABuffer = reinterpret_cast<fn_ES_DSP_LL_UnprepareDMABuffer>(dlsym(dsp_lib, "ES_DSP_LL_UnprepareDMABuffer"));
    g_syms.SubmitTask = reinterpret_cast<fn_ES_DSP_LL_SubmitTask>(dlsym(dsp_lib, "ES_DSP_LL_SubmitTask"));

    g_syms.MemAlloc = reinterpret_cast<fn_ES_SYS_MemAlloc>(dlsym(mem_lib, "ES_SYS_MemAlloc"));
    g_syms.MemFree = reinterpret_cast<fn_ES_SYS_MemFree>(dlsym(mem_lib, "ES_SYS_MemFree"));
    g_syms.Mmap = reinterpret_cast<fn_ES_SYS_Mmap>(dlsym(mem_lib, "ES_SYS_Mmap"));
    g_syms.Munmap = reinterpret_cast<fn_ES_SYS_Munmap>(dlsym(mem_lib, "ES_SYS_Munmap"));

    g_syms_loaded = true;
}

} // namespace

DspContext::DspContext(DspCore core)
    : core_(core), dsp_fd_(-1) {
    init_dsp_symbols();
}

DspContext::~DspContext() {
    close();
}

void DspContext::open() {
    if (is_open()) return;
    int fd = -1;
    int ret = g_syms.Open(static_cast<int>(core_), &fd);
    if (ret != 0 || fd < 0) {
        std::ostringstream oss;
        oss << "ES_DSP_LL_Open failed for core " << static_cast<int>(core_)
            << " with code 0x" << std::hex << ret;
        throw std::runtime_error(oss.str());
    }
    dsp_fd_ = fd;
}

void DspContext::close() {
    if (is_open()) {
        g_syms.Close(dsp_fd_);
        dsp_fd_ = -1;
    }
}

uint32_t DspContext::load_operator(const std::string& op_name, const std::string& op_dir) {
    if (!is_open()) open();
    uint32_t handle = 0;
    int ret = g_syms.LoadOperator(dsp_fd_, op_name.c_str(), op_dir.c_str(), &handle);
    if (ret != 0) {
        std::ostringstream oss;
        oss << "ES_DSP_LL_LoadOperator('" << op_name << "') failed with code 0x"
            << std::hex << ret;
        throw std::runtime_error(oss.str());
    }
    return handle;
}

void DspContext::unload_operator(uint32_t handle) {
    if (!is_open() || handle == 0) return;
    g_syms.UnloadOperator(dsp_fd_, handle);
}

DspBuffer DspContext::allocate_buffer(size_t size) {
    DspBuffer buf;
    buf.size = size;
    buf.alloc_size = ALIGN_2M(size);
    buf.offset = 0;

    int ret = g_syms.MemAlloc(&buf.mem_fd, 0, "dsp_runtime", nullptr, buf.alloc_size);
    if (ret != 0) {
        throw std::runtime_error("ES_SYS_MemAlloc failed for DSP buffer");
    }

    buf.host_ptr = g_syms.Mmap(buf.mem_fd, buf.alloc_size, 0);
    if (!buf.host_ptr) {
        g_syms.MemFree(buf.mem_fd);
        throw std::runtime_error("ES_SYS_Mmap failed for DSP buffer");
    }

    return buf;
}

void DspContext::free_buffer(DspBuffer& buf) {
    if (buf.host_ptr && buf.alloc_size > 0) {
        g_syms.Munmap(buf.host_ptr, buf.alloc_size);
        buf.host_ptr = nullptr;
    }
    if (buf.mem_fd != 0) {
        g_syms.MemFree(buf.mem_fd);
        buf.mem_fd = 0;
    }
}

void DspContext::prepare_dma(const DspBuffer& buf) {
    if (!is_open()) open();
    RawDevBuf dev_buf;
    dev_buf.memFd = buf.mem_fd;
    dev_buf.offset = buf.offset;
    dev_buf.size = buf.size;
    dev_buf.reserved = 0;

    int ret = g_syms.PrepareDMABuffer(dsp_fd_, dev_buf);
    if (ret != 0) {
        std::ostringstream oss;
        oss << "ES_DSP_LL_PrepareDMABuffer failed with code 0x" << std::hex << ret;
        throw std::runtime_error(oss.str());
    }
}

void DspContext::unprepare_dma(const DspBuffer& buf) {
    if (!is_open()) return;
    g_syms.UnprepareDMABuffer(dsp_fd_, buf.mem_fd);
}

void DspContext::submit_task(
    uint32_t op_handle,
    const std::vector<DspBuffer>& cfgs,
    const std::vector<DspBuffer>& inputs,
    const std::vector<DspBuffer>& outputs,
    bool sync_cache
) {
    if (!is_open()) open();

    RawDspTask task;
    std::memset(&task, 0, sizeof(task));
    task.operatorHandle = op_handle;
    task.bufferCntCfg = static_cast<uint32_t>(cfgs.size());
    task.bufferCntInput = static_cast<uint32_t>(inputs.size());
    task.bufferCntOutput = static_cast<uint32_t>(outputs.size());
    task.syncCache = sync_cache ? 1 : 0;
    task.pollMode = 0;

    int idx = 0;
    for (const auto& b : cfgs) {
        task.dspBuffers[idx++] = {b.mem_fd, b.offset, b.size, 0};
    }
    for (const auto& b : inputs) {
        task.dspBuffers[idx++] = {b.mem_fd, b.offset, b.size, 0};
    }
    for (const auto& b : outputs) {
        task.dspBuffers[idx++] = {b.mem_fd, b.offset, b.size, 0};
    }

    int ret = g_syms.SubmitTask(dsp_fd_, &task);
    if (ret != 0) {
        std::ostringstream oss;
        oss << "ES_DSP_LL_SubmitTask failed with code 0x" << std::hex << ret;
        throw std::runtime_error(oss.str());
    }
}

std::string DspContext::get_version_string() {
    char buf[128] = {0};
    if (g_syms.GetVersionString) {
        g_syms.GetVersionString(buf, sizeof(buf));
    }
    return std::string(buf);
}

uint64_t DspContext::get_version() {
    uint64_t ver = 0;
    if (g_syms.GetVersion) {
        g_syms.GetVersion(&ver);
    }
    return ver;
}

} // namespace eswin
