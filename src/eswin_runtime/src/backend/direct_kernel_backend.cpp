/**
 * @file direct_kernel_backend.cpp
 * @brief Layer 0' implementation: dma-heap allocation and /dev/npu* ioctls (Milestone 11).
 *
 * Nothing here calls dlopen. Allocation goes through the standard Linux dma-heap uAPI on
 * /dev/dma_heap/mmz_nid_<die>_part_0, mapping through mmap, and SMMU binding through the
 * eic7700_npu.ko ioctl dispatch table. That is what makes the path blob-free.
 *
 * Cache maintenance uses the standard DMA_BUF_IOCTL_SYNC rather than the vendor
 * ES_SYS_MemFlushCache, so CPU writes are published to the device the same way on any
 * dma-buf exporter.
 */

#include "direct_kernel_backend.hpp"

#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <sstream>

namespace eswin {

namespace {

constexpr size_t kPageAlign = 4096;

// Linux dma-buf synchronization uAPI (dma-buf.h). Magic is 'H', same letter as dma-heap
// but a different command number.
constexpr uint32_t kDmaBufIoctlMagic = 0x48;
struct DmaBufSync {
    uint64_t flags;
};
static_assert(sizeof(DmaBufSync) == 8, "dma_buf_sync layout");

constexpr uint64_t kDmaBufSyncRead  = 1;
constexpr uint64_t kDmaBufSyncWrite = 2;
constexpr uint64_t kDmaBufSyncStart = 0;
constexpr uint64_t kDmaBufSyncEnd   = (1ull << 4);

// _IOW('H', 0, struct dma_buf_sync): write-only from userspace, 8-byte payload.
constexpr uint32_t kDmaBufIoctlSync =
    (static_cast<uint32_t>(kIocWrite) << 30) |
    (static_cast<uint32_t>(sizeof(DmaBufSync)) << 16) |
    (static_cast<uint32_t>(kDmaBufIoctlMagic) << 8) | 0x00u;

size_t align_up(size_t n) {
    return (n + kPageAlign - 1) & ~(kPageAlign - 1);
}

} // namespace

const char *dma_heap_node(uint16_t die_id) {
    return (die_id == 1) ? "/dev/dma_heap/mmz_nid_1_part_0"
                         : "/dev/dma_heap/mmz_nid_0_part_0";
}

const char *npu_device_node(uint16_t die_id) {
    return (die_id == 1) ? "/dev/npu1" : "/dev/npu0";
}

void DirectDmaBuffer::release() {
    if (vaddr && vaddr != MAP_FAILED) {
        ::munmap(vaddr, size);
    }
    if (fd >= 0) {
        ::close(fd);
    }
    vaddr = nullptr;
    fd = -1;
    size = 0;
}

DirectKernelBackend::~DirectKernelBackend() {
    close();
}

int DirectKernelBackend::probe_device_count() {
    int count = 0;
    for (uint16_t die = 0; die < 2; ++die) {
        if (::access(npu_device_node(die), F_OK) == 0) ++count;
    }
    return count;
}

bool DirectKernelBackend::open(uint16_t die_id, std::string &err) {
    close();
    err.clear();

    if (die_id > 1) {
        err = "DirectKernelBackend: die_id must be 0 or 1";
        return false;
    }

    // O_RDWR: the driver needs write access for SMMU mapping. The nodes are 0660
    // root:root, so this fails with EACCES unless run as root or a udev rule relaxes
    // the mode - report that specifically, because it is the most common board failure.
    int fd = ::open(npu_device_node(die_id), O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        std::ostringstream os;
        os << "DirectKernelBackend: cannot open " << npu_device_node(die_id) << ": "
           << std::strerror(errno);
        if (errno == EACCES) {
            os << " (nodes are 0660 root:root - use sudo or add "
                  "/etc/udev/rules.d/99-npu.rules)";
        }
        err = os.str();
        return false;
    }

    m_fd = fd;
    m_die_id = die_id;
    return true;
}

void DirectKernelBackend::close() {
    if (m_fd >= 0) {
        ::close(m_fd);
    }
    m_fd = -1;
    m_die_id = 0;
}

bool DirectKernelBackend::alloc_dma(size_t bytes, uint16_t die_id, DirectDmaBuffer &out,
                                    std::string &err) {
    out.release();
    err.clear();

    if (bytes == 0) {
        err = "DirectKernelBackend: refusing a zero-byte allocation";
        return false;
    }
    const size_t len = align_up(bytes);

    int heap_fd = ::open(dma_heap_node(die_id), O_RDWR | O_CLOEXEC);
    if (heap_fd < 0) {
        std::ostringstream os;
        os << "DirectKernelBackend: cannot open dma-heap " << dma_heap_node(die_id)
           << ": " << std::strerror(errno);
        err = os.str();
        return false;
    }

    DmaHeapAllocationData req{};
    req.len = len;
    req.fd = 0;
    req.fd_flags = kDmaHeapFdCloexec;
    req.heap_flags = 0;

    if (::ioctl(heap_fd, kDmaHeapIoctlAlloc, &req) != 0) {
        std::ostringstream os;
        os << "DirectKernelBackend: DMA_HEAP_IOCTL_ALLOC of " << len << " bytes on "
           << dma_heap_node(die_id) << " failed: " << std::strerror(errno)
           << " (is the MMZ carveout large enough?)";
        err = os.str();
        ::close(heap_fd);
        return false;
    }
    ::close(heap_fd);   // the buffer fd outlives the heap fd

    void *va = ::mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_SHARED,
                      static_cast<int>(req.fd), 0);
    if (va == MAP_FAILED) {
        std::ostringstream os;
        os << "DirectKernelBackend: mmap of the dma-buf failed: " << std::strerror(errno);
        err = os.str();
        ::close(static_cast<int>(req.fd));
        return false;
    }

    out.fd = static_cast<int>(req.fd);
    out.vaddr = va;
    out.size = len;
    out.die_id = die_id;
    return true;
}

int DirectKernelBackend::prepare_dma(const DirectDmaBuffer &buf) {
    if (m_fd < 0 || buf.fd < 0) return -1;

    // The command word encodes a 4-byte payload (see header), so pass the buffer fd as a
    // uint32_t rather than the 32-byte vendor ES_DEV_BUF. If the kernel actually wants
    // the full struct this returns EINVAL, which is the signal to revisit the RE.
    uint32_t payload = static_cast<uint32_t>(buf.fd);
    return ::ioctl(m_fd, kNpuIoctlPrepareDma, &payload);
}

int DirectKernelBackend::unprepare_dma(const DirectDmaBuffer &buf) {
    if (m_fd < 0 || buf.fd < 0) return -1;
    uint32_t payload = static_cast<uint32_t>(buf.fd);
    return ::ioctl(m_fd, kNpuIoctlUnprepareDma, &payload);
}

bool DirectKernelBackend::flush_cache(const DirectDmaBuffer &buf) {
    if (buf.fd < 0) return false;

    // Publish CPU writes to the device. START then END brackets the access window; the
    // exporter is free to no-op either when the heap is uncached.
    DmaBufSync start{kDmaBufSyncStart | kDmaBufSyncWrite};
    DmaBufSync end{kDmaBufSyncEnd | kDmaBufSyncWrite};
    if (::ioctl(buf.fd, kDmaBufIoctlSync, &start) != 0) return false;
    return ::ioctl(buf.fd, kDmaBufIoctlSync, &end) == 0;
}

} // namespace eswin
