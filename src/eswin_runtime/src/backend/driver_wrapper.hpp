/**
 * @file driver_wrapper.hpp
 * @brief Dynamic loader and hardware abstraction wrapper for ESWIN NPU and MMZ drivers.
 *
 * Encapsulates dynamic resolution of symbols from:
 *   - libedla_runtime.so.1.0 (NPU core execution, task scheduling, streams)
 *   - libes_memory.so        (Contiguous MMZ DMA buffer allocation and mmap)
 *   - libes_sys.so.1         (Subsystem lifecycle, inter-channel binding)
 */

#ifndef ES_DRIVER_WRAPPER_HPP
#define ES_DRIVER_WRAPPER_HPP

#include <cstdint>
#include <cstddef>
#include <string>
#include <memory>
#include <mutex>
#include <vector>
#include <iostream>

namespace eswin {

// =============================================================================
// Hardware Driver Struct Definitions (DWARF-verified from EIC7700X SDK)
// =============================================================================

#pragma pack(push, 8)
struct DevBuf {
    uint64_t mem_fd;    // DMA file descriptor / memory handle
    uint64_t offset;    // Byte offset into contiguous buffer
    uint64_t size;      // Allocation size in bytes
    uint64_t reserved;  // Alignment padding
};

struct NpuTask {
    uint32_t taskId;
    uint32_t modelId;
    uint8_t  inputFdNum;
    uint8_t  outputFdNum;
    uint8_t  reserved0[6]; // Alignment padding to offset 16
    DevBuf   inputFd[10];
    DevBuf   outputFd[10];
    void    *callback;
    void    *callbackArg;
    uint32_t state;
    uint8_t  sdkPrivate[64];
    uint8_t  reserved1[4]; // Padding to align total size to 744 bytes
};
#pragma pack(pop)

static_assert(sizeof(NpuTask) == 744, "NpuTask size mismatch with vendor NPU_TASK_S");
static_assert(offsetof(NpuTask, inputFd) == 16, "NpuTask inputFd offset mismatch");
static_assert(offsetof(NpuTask, outputFd) == 336, "NpuTask outputFd offset mismatch");
static_assert(offsetof(NpuTask, callback) == 656, "NpuTask callback offset mismatch");
static_assert(offsetof(NpuTask, state) == 672, "NpuTask state offset mismatch");

// Cache modes for DMA allocation
enum class CacheMode : int {
    NoCache = 0,
    Cached  = 1
};

// =============================================================================
// Vendor Function Pointer Types
// =============================================================================

// NPU Runtime APIs (libedla_runtime.so)
using fn_ES_NPU_GetNumDevices       = int (*)(uint16_t *deviceNum);
using fn_ES_NPU_SetDevice           = int (*)(uint16_t deviceId);
using fn_ES_NPU_ReleaseDevice       = int (*)(uint16_t deviceId);
using fn_ES_NPU_GetDevice           = int (*)(uint16_t *deviceId);
using fn_ES_NPU_CreateContext       = int (*)(void **context, uint16_t deviceId);
using fn_ES_NPU_DestroyContext      = int (*)(void *context);
using fn_ES_NPU_CreateStream        = int (*)(void **stream);
using fn_ES_NPU_DestroyStream       = int (*)(void *stream);
using fn_ES_NPU_SynchronizeStream   = int (*)(void *stream);
using fn_ES_NPU_LoadModelFromFile   = int (*)(uint32_t *modelId, const char *modelPath);
using fn_ES_NPU_UnloadModel         = int (*)(uint32_t modelId);
using fn_ES_NPU_GetNumInputTensors  = int (*)(uint32_t modelId, uint32_t *tensorNum);
using fn_ES_NPU_GetNumOutputTensors = int (*)(uint32_t modelId, uint32_t *tensorNum);
using fn_ES_NPU_GetInputTensorDesc  = int (*)(uint32_t modelId, int32_t tensorId, void *tensor);
using fn_ES_NPU_GetOutputTensorDesc = int (*)(uint32_t modelId, int32_t tensorId, void *tensor);
using fn_ES_NPU_Prepare             = int (*)(DevBuf *devBuf);
using fn_ES_NPU_Unprepare           = int (*)(DevBuf *devBuf);
using fn_ES_NPU_Submit              = int (*)(NpuTask *tasks, uint32_t numTasks);
using fn_ES_NPU_SubmitFlexible      = int (*)(NpuTask *tasks, uint32_t numTasks, void *stream);

struct NpuTensorDesc {
    char     name[81];
    uint8_t  _pad0[7];
    uint64_t buffer_size;
    int32_t  dims[4];
    uint8_t  data_format;
    uint8_t  data_type;
    uint8_t  data_category;
    uint8_t  pixel_format;
    uint8_t  pixel_mapping;
    uint8_t  _pad1[3];
    uint32_t stride[8];
};

// Memory & System APIs (libes_memory.so & libes_sys.so.1)
using fn_ES_SYS_Init              = int (*)(void);
using fn_ES_SYS_Exit              = int (*)(void);
using fn_ES_SYS_MemAlloc          = int (*)(uint64_t *memFd, int cacheType, const char *pTag, const char *pZone, uint64_t len);
using fn_ES_SYS_MemFree           = int (*)(uint64_t memFd);
using fn_ES_SYS_Mmap              = void* (*)(uint64_t memFd, uint64_t size, int cacheType);
using fn_ES_SYS_Munmap            = int (*)(void *pVirAddr, uint64_t size);
using fn_ES_SYS_MemFlushCache     = int (*)(uint64_t memFd);
using fn_ES_SYS_RcvMemFd          = int (*)(uint64_t sndMemFd, uint64_t *rcvdMemFd);

// =============================================================================
// Driver Wrapper Class
// =============================================================================

class EswinDriver {
public:
    static EswinDriver& instance();

    ~EswinDriver();

    // Lifecycle
    bool initialize();
    void shutdown();
    bool is_hardware_available() const { return m_hardware_available; }

    // Device management
    uint16_t get_num_devices() const { return m_num_devices; }
    int set_device(uint16_t device_id);
    int release_device(uint16_t device_id);

    // Context & Streams
    int create_context(void **context, uint16_t device_id);
    int destroy_context(void *context);
    int create_stream(void **stream);
    int destroy_stream(void *stream);
    int sync_stream(void *stream);

    // Model management
    int load_model(uint32_t *model_id, const std::string &path);
    int unload_model(uint32_t model_id);
    int get_num_input_tensors(uint32_t model_id, uint32_t *num);
    int get_num_output_tensors(uint32_t model_id, uint32_t *num);
    int get_input_tensor_desc(uint32_t model_id, int32_t tensor_id, NpuTensorDesc *desc);
    int get_output_tensor_desc(uint32_t model_id, int32_t tensor_id, NpuTensorDesc *desc);

    // DMA & Memory operations
    int mem_alloc(uint64_t *mem_fd, size_t size, CacheMode cache_mode, const char *zone = nullptr);
    int mem_free(uint64_t mem_fd);
    int mem_rcv_fd(uint64_t snd_mem_fd, uint64_t *rcvd_mem_fd);
    void* mem_mmap(uint64_t mem_fd, size_t size, CacheMode cache_mode);
    int mem_munmap(void *vaddr, size_t size);
    int mem_flush_cache(uint64_t mem_fd);

    // Task & SMMU operations
    int prepare_dev_buf(DevBuf *buf);
    int unprepare_dev_buf(DevBuf *buf);
    int submit_task(NpuTask *task, uint32_t num_tasks = 1);
    int submit_flexible_task(NpuTask *tasks, uint32_t num_tasks, void *stream);

    // Error decoding
    static std::string decode_error(int error_code);

private:
    EswinDriver();
    EswinDriver(const EswinDriver&) = delete;
    EswinDriver& operator=(const EswinDriver&) = delete;

    bool load_libraries();
    void* try_dlopen(const std::vector<std::string> &paths);

    void *m_lib_npu = nullptr;
    void *m_lib_mem = nullptr;
    void *m_lib_sys = nullptr;

    bool m_initialized = false;
    bool m_hardware_available = false;
    uint16_t m_num_devices = 0;
    std::mutex m_mutex;

    // Function pointers
    fn_ES_NPU_GetNumDevices     m_pGetNumDevices = nullptr;
    fn_ES_NPU_SetDevice         m_pSetDevice = nullptr;
    fn_ES_NPU_ReleaseDevice     m_pReleaseDevice = nullptr;
    fn_ES_NPU_GetDevice         m_pGetDevice = nullptr;
    fn_ES_NPU_CreateContext     m_pCreateContext = nullptr;
    fn_ES_NPU_DestroyContext    m_pDestroyContext = nullptr;
    fn_ES_NPU_CreateStream      m_pCreateStream = nullptr;
    fn_ES_NPU_DestroyStream     m_pDestroyStream = nullptr;
    fn_ES_NPU_SynchronizeStream m_pSyncStream = nullptr;
    fn_ES_NPU_LoadModelFromFile m_pLoadModel = nullptr;
    fn_ES_NPU_UnloadModel       m_pUnloadModel = nullptr;
    fn_ES_NPU_GetNumInputTensors  m_pGetNumInputTensors = nullptr;
    fn_ES_NPU_GetNumOutputTensors m_pGetNumOutputTensors = nullptr;
    fn_ES_NPU_GetInputTensorDesc  m_pGetInputTensorDesc = nullptr;
    fn_ES_NPU_GetOutputTensorDesc m_pGetOutputTensorDesc = nullptr;
    fn_ES_NPU_Prepare           m_pPrepare = nullptr;
    fn_ES_NPU_Unprepare         m_pUnprepare = nullptr;
    fn_ES_NPU_Submit            m_pSubmit = nullptr;
    fn_ES_NPU_SubmitFlexible    m_pSubmitFlexible = nullptr;

    fn_ES_SYS_Init              m_pSysInit = nullptr;
    fn_ES_SYS_Exit              m_pSysExit = nullptr;
    fn_ES_SYS_MemAlloc          m_pMemAlloc = nullptr;
    fn_ES_SYS_MemFree           m_pMemFree = nullptr;
    fn_ES_SYS_Mmap              m_pMmap = nullptr;
    fn_ES_SYS_Munmap            m_pMunmap = nullptr;
    fn_ES_SYS_MemFlushCache     m_pMemFlushCache = nullptr;
    fn_ES_SYS_RcvMemFd          m_pRcvMemFd = nullptr;
};

} // namespace eswin

#endif // ES_DRIVER_WRAPPER_HPP
