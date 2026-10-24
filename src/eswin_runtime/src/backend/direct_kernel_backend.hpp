/**
 * @file direct_kernel_backend.hpp
 * @brief Layer 0': direct kernel ioctl backend, no vendor userspace blob (Milestone 11).
 *
 * Goal: talk to /dev/npu0 and /dev/npu1 and to the standard Linux dma-heap uAPI directly,
 * so inference does not require libedla_runtime.so.1.0 or libes_memory.so.
 *
 * ## ioctl encodings are pinned to the observed values, not derived
 *
 * The catalog in findings/CPU_NPU_PROTOCOL.md was recovered by disassembling
 * npu_dev_ioctl in eic7700_npu.ko. Those numbers do NOT all match what _IOWR/_IOR would
 * generate from the struct the vendor passes, and the difference is load-bearing:
 *
 *   NPU_IOCTL_PREPARE_DMA is documented as _IOWR('n', 0x0F, struct ES_DEV_BUF)
 *   but its encoding is 0xc0046e0f, whose size field (bits 16..29) is 4.
 *   sizeof(ES_DEV_BUF) is 32 (DevBuf: mem_fd/offset/size/reserved), which would encode
 *   as 0xc0206e0f. The kernel dispatches on the full 32-bit command word, so passing
 *   _IOWR('n', 0x0F, DevBuf) would be rejected as an unknown ioctl.
 *
 * The size-4 field is consistent with the mutex commands, which the same catalog types as
 * uint32_t. The most likely reading is that the kernel takes a 4-byte handle (the DMA-BUF
 * fd) and the vendor's ES_DEV_BUF annotation describes the userspace struct, not the
 * ioctl payload. That is a hypothesis, not a verified fact - see the "Open questions"
 * section of docs/DIRECT_KERNEL_IOCTL_SPEC.md.
 *
 * So the constants below are the literal observed values, each with a static_assert on
 * its decoded direction/type/nr/size fields. The asserts run on any host, which makes the
 * encoding table itself testable without the board.
 */

#ifndef ES_DIRECT_KERNEL_BACKEND_HPP
#define ES_DIRECT_KERNEL_BACKEND_HPP

#include <cstddef>
#include <cstdint>
#include <string>

namespace eswin {

// --- Decoded ioctl command word helpers (Linux asm-generic/ioctl.h layout) ------
constexpr uint32_t ioctl_dir(uint32_t cmd)  { return cmd >> 30; }
constexpr uint32_t ioctl_size(uint32_t cmd) { return (cmd >> 16) & 0x3FFF; }
constexpr uint32_t ioctl_type(uint32_t cmd) { return (cmd >> 8) & 0xFF; }
constexpr uint32_t ioctl_nr(uint32_t cmd)   { return cmd & 0xFF; }

constexpr uint32_t kIocNone  = 0;
constexpr uint32_t kIocWrite = 1;   // userspace -> kernel
constexpr uint32_t kIocRead  = 2;   // kernel -> userspace
constexpr uint32_t kIocRW    = kIocWrite | kIocRead;

constexpr uint32_t kNpuIoctlMagic = 0x6E;   // 'n'

// --- eic7700_npu.ko dispatch table (observed encodings) ------------------------
constexpr uint32_t kNpuIoctlSubmitModel   = 0xc0046e03u;
constexpr uint32_t kNpuIoctlUnloadModel   = 0xc0046e04u;
constexpr uint32_t kNpuIoctlSubmitTask    = 0xc0046e05u;
constexpr uint32_t kNpuIoctlProcessReport = 0x80046e08u;
constexpr uint32_t kNpuIoctlSubmitOpTask  = 0xc0046e09u;
constexpr uint32_t kNpuIoctlGetSram       = 0x80046e0au;
constexpr uint32_t kNpuIoctlHandlePerf    = 0x80046e0bu;
constexpr uint32_t kNpuIoctlGetPerfData   = 0x80046e0cu;
constexpr uint32_t kNpuIoctlMutexLock     = 0x80046e0du;
constexpr uint32_t kNpuIoctlMutexUnlock   = 0xc0046e0eu;
constexpr uint32_t kNpuIoctlPrepareDma    = 0xc0046e0fu;
constexpr uint32_t kNpuIoctlUnprepareDma  = 0xc0046e10u;
constexpr uint32_t kNpuIoctlMutexTrylock  = 0xc0046e11u;

// Every command shares magic 'n'. The size field is 4 for all of them, which is the
// discrepancy documented above.
static_assert(ioctl_type(kNpuIoctlSubmitModel) == kNpuIoctlMagic, "magic");
static_assert(ioctl_type(kNpuIoctlPrepareDma) == kNpuIoctlMagic, "magic");
static_assert(ioctl_type(kNpuIoctlMutexTrylock) == kNpuIoctlMagic, "magic");
static_assert(ioctl_nr(kNpuIoctlSubmitModel) == 0x03, "nr");
static_assert(ioctl_nr(kNpuIoctlSubmitTask) == 0x05, "nr");
static_assert(ioctl_nr(kNpuIoctlPrepareDma) == 0x0f, "nr");
static_assert(ioctl_nr(kNpuIoctlUnprepareDma) == 0x10, "nr");
static_assert(ioctl_nr(kNpuIoctlMutexTrylock) == 0x11, "nr");
static_assert(ioctl_dir(kNpuIoctlSubmitModel) == kIocRW, "SUBMIT_MODEL is _IOWR");
static_assert(ioctl_dir(kNpuIoctlProcessReport) == kIocRead, "PROCESS_REPORT is _IOR");
static_assert(ioctl_dir(kNpuIoctlMutexLock) == kIocRead, "MUTEX_LOCK is _IOR");
static_assert(ioctl_dir(kNpuIoctlMutexUnlock) == kIocRW, "MUTEX_UNLOCK is _IOWR");
static_assert(ioctl_size(kNpuIoctlPrepareDma) == 4,
              "PREPARE_DMA encodes a 4-byte payload, not sizeof(ES_DEV_BUF)==32");
static_assert(ioctl_size(kNpuIoctlSubmitTask) == 4, "SUBMIT_TASK payload size");

// --- Standard Linux dma-heap uAPI (blob-free allocation) -----------------------
constexpr uint32_t kDmaHeapMagic = 0x48;   // 'H'
// struct dma_heap_allocation_data { u64 len; u32 fd; u32 fd_flags; u64 heap_flags; }
struct DmaHeapAllocationData {
    uint64_t len;
    uint32_t fd;
    uint32_t fd_flags;
    uint64_t heap_flags;
};
static_assert(sizeof(DmaHeapAllocationData) == 24, "dma_heap_allocation_data layout");

// _IOWR('H', 0x00, struct dma_heap_allocation_data)
constexpr uint32_t kDmaHeapIoctlAlloc =
    (static_cast<uint32_t>(kIocRW) << 30) |
    (static_cast<uint32_t>(sizeof(DmaHeapAllocationData)) << 16) |
    (static_cast<uint32_t>(kDmaHeapMagic) << 8) | 0x00u;
static_assert(kDmaHeapIoctlAlloc == 0xc0184800u, "DMA_HEAP_IOCTL_ALLOC encoding");

constexpr uint32_t kDmaHeapFdCloexec = 1;  // O_CLOEXEC for the returned buffer fd
constexpr uint32_t kDmaHeapFdShared  = 2;  // O_SHARED for cross-process import

/// Per-die MMZ dma-heap device nodes.
const char *dma_heap_node(uint16_t die_id);
/// NPU character device node for a die.
const char *npu_device_node(uint16_t die_id);

/**
 * @brief One physically contiguous buffer allocated straight from a dma-heap.
 *
 * Owns the buffer fd and its mapping; released on destruction. This replaces
 * ES_SYS_MemAlloc + ES_SYS_Mmap for the blob-free path.
 */
struct DirectDmaBuffer {
    int      fd = -1;
    void    *vaddr = nullptr;
    size_t   size = 0;
    uint16_t die_id = 0;

    bool valid() const noexcept { return fd >= 0; }
    void release();
};

class DirectKernelBackend {
public:
    DirectKernelBackend() = default;
    ~DirectKernelBackend();

    DirectKernelBackend(const DirectKernelBackend &) = delete;
    DirectKernelBackend &operator=(const DirectKernelBackend &) = delete;

    /// True when /dev/npu<die> exists and could be opened.
    bool open(uint16_t die_id, std::string &err);
    void close();
    bool is_open() const noexcept { return m_fd >= 0; }
    uint16_t die_id() const noexcept { return m_die_id; }
    int fd() const noexcept { return m_fd; }

    /// Number of /dev/npu* nodes present on this machine (0 off-board).
    static int probe_device_count();

    /// Allocates from the die's dma-heap and maps it. No vendor library involved.
    bool alloc_dma(size_t bytes, uint16_t die_id, DirectDmaBuffer &out, std::string &err);

    /**
     * @brief Maps a buffer into the NPU's SMMU (NPU_IOCTL_PREPARE_DMA).
     *
     * Passes the 4-byte payload the ioctl encoding implies, i.e. the buffer fd. See the
     * header note: the vendor struct is 32 bytes but the command word says 4.
     */
    int prepare_dma(const DirectDmaBuffer &buf);
    int unprepare_dma(const DirectDmaBuffer &buf);

    /// Cache maintenance for a CPU-written buffer before the NPU reads it.
    bool flush_cache(const DirectDmaBuffer &buf);

private:
    int      m_fd = -1;
    uint16_t m_die_id = 0;
};

} // namespace eswin

#endif // ES_DIRECT_KERNEL_BACKEND_HPP
