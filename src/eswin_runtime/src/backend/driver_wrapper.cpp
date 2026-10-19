/**
 * @file driver_wrapper.cpp
 * @brief Dynamic loader implementation and error decoder for ESWIN NPU/MMZ drivers.
 */

#include "driver_wrapper.hpp"

#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <iomanip>

namespace eswin {

EswinDriver& EswinDriver::instance() {
    static EswinDriver s_instance;
    return s_instance;
}

EswinDriver::EswinDriver() = default;

EswinDriver::~EswinDriver() {
    shutdown();
}

void* EswinDriver::try_dlopen(const std::vector<std::string> &paths) {
    for (const auto &path : paths) {
        void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (handle) {
            return handle;
        }
    }
    return nullptr;
}

bool EswinDriver::load_libraries() {
    // 1. Load NPU Runtime (libedla_runtime.so)
    const std::vector<std::string> npu_paths = {
        "/lib/libedla_runtime.so.1.0",
        "/usr/lib/libedla_runtime.so.1.0",
        "/opt/eswin/lib/libedla_runtime.so.1.0",
        "libedla_runtime.so.1.0",
        "libedla_runtime.so"
    };
    m_lib_npu = try_dlopen(npu_paths);

    // 2. Load Memory Driver (libes_memory.so)
    const std::vector<std::string> mem_paths = {
        "/lib/libes_memory.so",
        "/usr/lib/libes_memory.so",
        "/opt/eswin/lib/libes_memory.so",
        "libes_memory.so"
    };
    m_lib_mem = try_dlopen(mem_paths);

    // 3. Load System Services (libes_sys.so.1)
    const std::vector<std::string> sys_paths = {
        "/lib/libes_sys.so.1",
        "/usr/lib/libes_sys.so.1",
        "/opt/eswin/lib/libes_sys.so.1",
        "libes_sys.so.1",
        "libes_sys.so"
    };
    m_lib_sys = try_dlopen(sys_paths);

    if (!m_lib_npu || !m_lib_mem) {
        return false;
    }

    // Resolve NPU functions
    m_pGetNumDevices  = (fn_ES_NPU_GetNumDevices)dlsym(m_lib_npu, "ES_NPU_GetNumDevices");
    m_pSetDevice      = (fn_ES_NPU_SetDevice)dlsym(m_lib_npu, "ES_NPU_SetDevice");
    m_pReleaseDevice  = (fn_ES_NPU_ReleaseDevice)dlsym(m_lib_npu, "ES_NPU_ReleaseDevice");
    m_pGetDevice      = (fn_ES_NPU_GetDevice)dlsym(m_lib_npu, "ES_NPU_GetDevice");
    m_pCreateContext  = (fn_ES_NPU_CreateContext)dlsym(m_lib_npu, "ES_NPU_CreateContext");
    m_pDestroyContext = (fn_ES_NPU_DestroyContext)dlsym(m_lib_npu, "ES_NPU_DestroyContext");
    m_pCreateStream   = (fn_ES_NPU_CreateStream)dlsym(m_lib_npu, "ES_NPU_CreateStream");
    m_pDestroyStream  = (fn_ES_NPU_DestroyStream)dlsym(m_lib_npu, "ES_NPU_DestroyStream");
    m_pSyncStream     = (fn_ES_NPU_SynchronizeStream)dlsym(m_lib_npu, "ES_NPU_SynchronizeStream");
    m_pLoadModel      = (fn_ES_NPU_LoadModelFromFile)dlsym(m_lib_npu, "ES_NPU_LoadModelFromFile");
    m_pUnloadModel    = (fn_ES_NPU_UnloadModel)dlsym(m_lib_npu, "ES_NPU_UnloadModel");
    m_pGetNumInputTensors  = (fn_ES_NPU_GetNumInputTensors)dlsym(m_lib_npu, "ES_NPU_GetNumInputTensors");
    m_pGetNumOutputTensors = (fn_ES_NPU_GetNumOutputTensors)dlsym(m_lib_npu, "ES_NPU_GetNumOutputTensors");
    m_pGetInputTensorDesc  = (fn_ES_NPU_GetInputTensorDesc)dlsym(m_lib_npu, "ES_NPU_GetInputTensorDesc");
    m_pGetOutputTensorDesc = (fn_ES_NPU_GetOutputTensorDesc)dlsym(m_lib_npu, "ES_NPU_GetOutputTensorDesc");
    m_pPrepare        = (fn_ES_NPU_Prepare)dlsym(m_lib_npu, "ES_NPU_Prepare");
    m_pUnprepare      = (fn_ES_NPU_Unprepare)dlsym(m_lib_npu, "ES_NPU_Unprepare");
    m_pSubmit         = (fn_ES_NPU_Submit)dlsym(m_lib_npu, "ES_NPU_Submit");
    m_pSubmitFlexible = (fn_ES_NPU_SubmitFlexible)dlsym(m_lib_npu, "ES_NPU_SubmitFlexibleTask");

    // Resolve Memory functions
    m_pMemAlloc       = (fn_ES_SYS_MemAlloc)dlsym(m_lib_mem, "ES_SYS_MemAlloc");
    m_pMemFree        = (fn_ES_SYS_MemFree)dlsym(m_lib_mem, "ES_SYS_MemFree");
    m_pMmap           = (fn_ES_SYS_Mmap)dlsym(m_lib_mem, "ES_SYS_Mmap");
    m_pMunmap         = (fn_ES_SYS_Munmap)dlsym(m_lib_mem, "ES_SYS_Munmap");
    m_pMemFlushCache  = (fn_ES_SYS_MemFlushCache)dlsym(m_lib_mem, "ES_SYS_MemFlushCache");
    m_pRcvMemFd       = (fn_ES_SYS_RcvMemFd)dlsym(m_lib_mem, "ES_SYS_RcvMemFd");

    // Resolve System functions if available
    if (m_lib_sys) {
        m_pSysInit = (fn_ES_SYS_Init)dlsym(m_lib_sys, "ES_SYS_Init");
        m_pSysExit = (fn_ES_SYS_Exit)dlsym(m_lib_sys, "ES_SYS_Exit");
    }

    return (m_pGetNumDevices && m_pSetDevice && m_pLoadModel && m_pMemAlloc && m_pMmap);
}

bool EswinDriver::initialize() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_initialized) {
        return true;
    }

    if (!load_libraries()) {
        m_hardware_available = false;
        m_initialized = true;
        return false;
    }

    // 1. Initialize subsystem if ES_SYS_Init is present
    if (m_pSysInit) {
        m_pSysInit();
    }

    // 2. Query device count (CRITICAL: initializes internal device manager table)
    uint16_t count = 0;
    if (m_pGetNumDevices && m_pGetNumDevices(&count) == 0) {
        m_num_devices = count;
        m_hardware_available = (m_num_devices > 0);
    } else {
        m_hardware_available = false;
    }

    m_initialized = true;
    return m_hardware_available;
}

void EswinDriver::shutdown() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_initialized) return;

    if (m_pSysExit) {
        m_pSysExit();
    }

    if (m_lib_npu) { dlclose(m_lib_npu); m_lib_npu = nullptr; }
    if (m_lib_mem) { dlclose(m_lib_mem); m_lib_mem = nullptr; }
    if (m_lib_sys) { dlclose(m_lib_sys); m_lib_sys = nullptr; }

    m_initialized = false;
    m_hardware_available = false;
    m_num_devices = 0;
}

int EswinDriver::set_device(uint16_t device_id) {
    if (!m_hardware_available || !m_pSetDevice) return -1;
    return m_pSetDevice(device_id);
}

int EswinDriver::release_device(uint16_t device_id) {
    if (!m_hardware_available || !m_pReleaseDevice) return -1;
    return m_pReleaseDevice(device_id);
}

int EswinDriver::create_context(void **context, uint16_t device_id) {
    if (!m_hardware_available || !m_pCreateContext) return -1;
    return m_pCreateContext(context, device_id);
}

int EswinDriver::destroy_context(void *context) {
    if (!m_hardware_available || !m_pDestroyContext) return -1;
    return m_pDestroyContext(context);
}

int EswinDriver::create_stream(void **stream) {
    if (!m_hardware_available || !m_pCreateStream) return -1;
    return m_pCreateStream(stream);
}

int EswinDriver::destroy_stream(void *stream) {
    if (!m_hardware_available || !m_pDestroyStream) return -1;
    return m_pDestroyStream(stream);
}

int EswinDriver::sync_stream(void *stream) {
    if (!m_hardware_available || !m_pSyncStream) return -1;
    return m_pSyncStream(stream);
}

int EswinDriver::load_model(uint32_t *model_id, const std::string &path) {
    if (!m_hardware_available || !m_pLoadModel) return -1;
    return m_pLoadModel(model_id, path.c_str());
}

int EswinDriver::unload_model(uint32_t model_id) {
    if (!m_hardware_available || !m_pUnloadModel) return -1;
    return m_pUnloadModel(model_id);
}

int EswinDriver::get_num_input_tensors(uint32_t model_id, uint32_t *num) {
    if (!m_hardware_available || !m_pGetNumInputTensors) return -1;
    return m_pGetNumInputTensors(model_id, num);
}

int EswinDriver::get_num_output_tensors(uint32_t model_id, uint32_t *num) {
    if (!m_hardware_available || !m_pGetNumOutputTensors) return -1;
    return m_pGetNumOutputTensors(model_id, num);
}

int EswinDriver::get_input_tensor_desc(uint32_t model_id, int32_t tensor_id, NpuTensorDesc *desc) {
    if (!m_hardware_available || !m_pGetInputTensorDesc) return -1;
    return m_pGetInputTensorDesc(model_id, tensor_id, desc);
}

int EswinDriver::get_output_tensor_desc(uint32_t model_id, int32_t tensor_id, NpuTensorDesc *desc) {
    if (!m_hardware_available || !m_pGetOutputTensorDesc) return -1;
    return m_pGetOutputTensorDesc(model_id, tensor_id, desc);
}

int EswinDriver::mem_alloc(uint64_t *mem_fd, size_t size, CacheMode cache_mode, const char *zone) {
    if (!m_hardware_available || !m_pMemAlloc) return -1;
    return m_pMemAlloc(mem_fd, static_cast<int>(cache_mode), "es_runtime", zone, size);
}

int EswinDriver::mem_free(uint64_t mem_fd) {
    if (!m_hardware_available || !m_pMemFree) return -1;
    return m_pMemFree(mem_fd);
}

int EswinDriver::mem_rcv_fd(uint64_t snd_mem_fd, uint64_t *rcvd_mem_fd) {
    if (!m_hardware_available || !m_pRcvMemFd) return -1;
    return m_pRcvMemFd(snd_mem_fd, rcvd_mem_fd);
}

void* EswinDriver::mem_mmap(uint64_t mem_fd, size_t size, CacheMode cache_mode) {
    if (!m_hardware_available || !m_pMmap) return nullptr;
    return m_pMmap(mem_fd, size, static_cast<int>(cache_mode));
}

int EswinDriver::mem_munmap(void *vaddr, size_t size) {
    if (!m_hardware_available || !m_pMunmap) return -1;
    return m_pMunmap(vaddr, size);
}

int EswinDriver::mem_flush_cache(uint64_t mem_fd) {
    if (!m_hardware_available || !m_pMemFlushCache) return -1;
    return m_pMemFlushCache(mem_fd);
}

int EswinDriver::prepare_dev_buf(DevBuf *buf) {
    if (!m_hardware_available || !m_pPrepare) return -1;
    return m_pPrepare(buf);
}

int EswinDriver::unprepare_dev_buf(DevBuf *buf) {
    if (!m_hardware_available || !m_pUnprepare) return -1;
    return m_pUnprepare(buf);
}

int EswinDriver::submit_task(NpuTask *task, uint32_t num_tasks) {
    if (!m_hardware_available || !m_pSubmit) return -1;
    return m_pSubmit(task, num_tasks);
}

int EswinDriver::submit_flexible_task(NpuTask *tasks, uint32_t num_tasks, void *stream) {
    if (!m_hardware_available || !m_pSubmitFlexible) return -1;
    return m_pSubmitFlexible(tasks, num_tasks, stream);
}

std::string EswinDriver::decode_error(int error_code) {
    if (error_code == 0) return "SUCCESS (0)";

    uint32_t err = static_cast<uint32_t>(error_code);
    uint32_t mod_id = (err >> 16) & 0xFF;
    uint32_t code   = err & 0x1FFF;

    std::ostringstream oss;
    oss << "0x" << std::hex << std::setw(8) << std::setfill('0') << err
        << " (Mod: " << std::dec << mod_id << ", ErrId: " << code << "): ";

    // Known specific ESWIN error mappings
    if (mod_id == 15) { // NPU Module
        switch (code) {
            case 3:  oss << "ES_NPU_ERROR_ILLEGAL_PARAM (Invalid model parameter or shape)"; break;
            case 6:  oss << "ES_NPU_ERROR_NULL_PTR (Null pointer passed to API)"; break;
            case 12: oss << "ES_NPU_ERROR_NOMEM (NPU internal memory allocation failure)"; break;
            case 18: oss << "ES_NPU_ERROR_BUSY (Device or engine is busy)"; break;
            case 64: oss << "ES_NPU_ERROR_NOT_INIT (NPU subsystem not initialized)"; break;
            case 69: oss << "ES_NPU_ERROR_DEV_NOT_FOUND (Device index not found)"; break;
            case 70: oss << "ES_NPU_ERROR_DEV_MULTI_SET (Die collision: process already bound a device)"; break;
            case 71: oss << "ES_NPU_ERROR_DEV_OPEN_FAILED (Failed to open /dev/npu*)"; break;
            case 72: oss << "ES_NPU_ERROR_MODEL_NOT_PRESENT (Model uninitialized or device context missing)"; break;
            case 73: oss << "ES_NPU_ERROR_MODEL_NOT_FOUND (Model ID not found)"; break;
            case 85: oss << "ES_NPU_ERROR_IOCTL_FAIL (Driver ioctl call failed)"; break;
            case 88: oss << "ES_NPU_ERROR_LOAD_OP_FAILED (DSP/NPU kernel loader failure)"; break;
            default: oss << "Unknown NPU Error"; break;
        }
    } else if (mod_id == 2) { // SYS Module
        switch (code) {
            case 12: oss << "ES_SYS_ERROR_NOMEM (MMZ out of contiguous DMA memory)"; break;
            case 17: oss << "ES_SYS_ERROR_BADADDR (Invalid memory address or unmapped DMA buffer)"; break;
            default: oss << "System/Memory Subsystem Error"; break;
        }
    } else {
        oss << "General ESWIN Subsystem Error";
    }

    return oss.str();
}

} // namespace eswin
