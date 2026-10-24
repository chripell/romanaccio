/**
 * @file test_direct_kernel_mode.cpp
 * @brief Milestone 11 test suite: direct kernel backend (zero vendor blob).
 *
 * Two things are verified here, and it matters that they are separate:
 *
 *  1. The ioctl encoding table. This is pure arithmetic against the values recovered by
 *     disassembling npu_dev_ioctl, so it is fully testable on any host with no board and
 *     no driver. A wrong constant would be rejected by the kernel as an unknown command.
 *
 *  2. Blob-free allocation. On a board this allocates from /dev/dma_heap/mmz_nid_*_part_0
 *     through the standard Linux dma-heap uAPI, maps it, writes and reads it back, and
 *     runs DMA_BUF_IOCTL_SYNC. Off-board every one of those calls must fail gracefully
 *     with a message naming the node, which is what the negative tests assert.
 *
 * SMMU binding (NPU_IOCTL_PREPARE_DMA) is reported, not asserted: the command word
 * encodes a 4-byte payload while the vendor passes a 32-byte ES_DEV_BUF, so what the
 * kernel actually wants is an open question documented in
 * docs/DIRECT_KERNEL_IOCTL_SPEC.md. Asserting success there would either be a lie or
 * would fail the suite for a reason that is not a code defect.
 */

#include "../src/backend/direct_kernel_backend.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <unistd.h>

#define TEST_ASSERT(cond, msg)                                                  \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::cerr << "\033[1;31m[FAIL]\033[0m " << msg << " (" << #cond     \
                      << ") at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

using namespace eswin;

namespace {

void test_ioctl_encoding_table() {
    std::cout << "[*] Verifying the eic7700_npu.ko ioctl encoding table..." << std::endl;

    struct Row { const char *name; uint32_t cmd; uint32_t dir; uint32_t nr; };
    const Row rows[] = {
        {"SUBMIT_MODEL",   kNpuIoctlSubmitModel,   kIocRW,   0x03},
        {"UNLOAD_MODEL",   kNpuIoctlUnloadModel,   kIocRW,   0x04},
        {"SUBMIT_TASK",    kNpuIoctlSubmitTask,    kIocRW,   0x05},
        {"PROCESS_REPORT", kNpuIoctlProcessReport, kIocRead, 0x08},
        {"SUBMIT_OP_TASK", kNpuIoctlSubmitOpTask,  kIocRW,   0x09},
        {"GET_SRAM",       kNpuIoctlGetSram,       kIocRead, 0x0a},
        {"HANDLE_PERF",    kNpuIoctlHandlePerf,    kIocRead, 0x0b},
        {"GET_PERF_DATA",  kNpuIoctlGetPerfData,   kIocRead, 0x0c},
        {"MUTEX_LOCK",     kNpuIoctlMutexLock,     kIocRead, 0x0d},
        {"MUTEX_UNLOCK",   kNpuIoctlMutexUnlock,   kIocRW,   0x0e},
        {"PREPARE_DMA",    kNpuIoctlPrepareDma,    kIocRW,   0x0f},
        {"UNPREPARE_DMA",  kNpuIoctlUnprepareDma,  kIocRW,   0x10},
        {"MUTEX_TRYLOCK",  kNpuIoctlMutexTrylock,  kIocRW,   0x11},
    };

    for (const auto &r : rows) {
        TEST_ASSERT(ioctl_type(r.cmd) == kNpuIoctlMagic,
                    std::string(r.name) + " does not use magic 'n'");
        TEST_ASSERT(ioctl_nr(r.cmd) == r.nr, std::string(r.name) + " command nr mismatch");
        TEST_ASSERT(ioctl_dir(r.cmd) == r.dir, std::string(r.name) + " direction mismatch");
        std::printf("    %-15s 0x%08x  dir=%u size=%u type='%c' nr=0x%02x\n",
                    r.name, r.cmd, ioctl_dir(r.cmd), ioctl_size(r.cmd),
                    static_cast<char>(ioctl_type(r.cmd)), ioctl_nr(r.cmd));
    }
    TEST_ASSERT(sizeof(rows) / sizeof(rows[0]) == 13, "expected all 13 catalogued ioctls");

    // The load-bearing discrepancy: every NPU command encodes a 4-byte payload, even the
    // ones whose vendor annotation names a 32-byte struct.
    for (const auto &r : rows) {
        TEST_ASSERT(ioctl_size(r.cmd) == 4,
                    std::string(r.name) + " payload size is not 4");
    }

    // dma-heap and dma-buf are standard Linux, so their encodings are derivable.
    TEST_ASSERT(kDmaHeapIoctlAlloc == 0xc0184800u, "DMA_HEAP_IOCTL_ALLOC encoding");
    TEST_ASSERT(sizeof(DmaHeapAllocationData) == 24, "dma_heap_allocation_data size");
    std::cout << "    DMA_HEAP_IOCTL_ALLOC = 0x" << std::hex << kDmaHeapIoctlAlloc
              << std::dec << " over a 24-byte request struct." << std::endl;
    std::cout << "    13 NPU commands verified; all encode a 4-byte payload." << std::endl;
}

void test_node_naming() {
    std::cout << "\n[*] Verifying device node naming..." << std::endl;
    TEST_ASSERT(std::strcmp(npu_device_node(0), "/dev/npu0") == 0, "npu0 node name");
    TEST_ASSERT(std::strcmp(npu_device_node(1), "/dev/npu1") == 0, "npu1 node name");
    TEST_ASSERT(std::strcmp(dma_heap_node(0), "/dev/dma_heap/mmz_nid_0_part_0") == 0,
                "die 0 heap name");
    TEST_ASSERT(std::strcmp(dma_heap_node(1), "/dev/dma_heap/mmz_nid_1_part_0") == 0,
                "die 1 heap name");
    std::cout << "    NPU and dma-heap node paths match findings/CPU_NPU_PROTOCOL.md."
              << std::endl;
}

void test_graceful_absence() {
    std::cout << "\n[*] Verifying graceful failure when the hardware is absent..."
              << std::endl;

    DirectKernelBackend be;
    std::string err;

    // Out-of-range die is rejected before any syscall.
    TEST_ASSERT(!be.open(7, err), "invalid die_id accepted");
    TEST_ASSERT(err.find("die_id") != std::string::npos, "unhelpful die_id error: " + err);
    TEST_ASSERT(!be.is_open(), "backend claims open after a failed open()");

    // A zero-byte allocation is refused rather than handed to the kernel.
    DirectDmaBuffer buf;
    TEST_ASSERT(!be.alloc_dma(0, 0, buf, err), "zero-byte allocation accepted");
    TEST_ASSERT(!buf.valid(), "buffer marked valid after a refused allocation");

    // prepare/unprepare/flush on an unopened backend must not crash or issue a syscall
    // against fd -1.
    TEST_ASSERT(be.prepare_dma(buf) == -1, "prepare_dma on a closed backend should fail");
    TEST_ASSERT(be.unprepare_dma(buf) == -1, "unprepare_dma on a closed backend should fail");
    TEST_ASSERT(!be.flush_cache(buf), "flush_cache on an invalid buffer should fail");

    if (DirectKernelBackend::probe_device_count() == 0) {
        TEST_ASSERT(!be.open(0, err), "open() succeeded with no /dev/npu0 present");
        TEST_ASSERT(err.find("/dev/npu0") != std::string::npos,
                    "error should name the missing node: " + err);
        TEST_ASSERT(!be.alloc_dma(4096, 0, buf, err), "alloc_dma succeeded with no heap");
        TEST_ASSERT(err.find("dma-heap") != std::string::npos,
                    "error should mention the dma-heap: " + err);
        std::cout << "    Off-board: open() and alloc_dma() both fail cleanly and name"
                     "\n    the missing node." << std::endl;
    } else {
        std::cout << "    Hardware present; absence tests skipped." << std::endl;
    }
}

void test_on_hardware() {
    const int dies = DirectKernelBackend::probe_device_count();
    std::cout << "\n[*] Hardware probe: " << dies << " NPU node(s) present." << std::endl;
    if (dies == 0) {
        std::cout << "    Off-board: dma-heap allocation and SMMU binding NOT exercised."
                  << std::endl;
        return;
    }

    DirectKernelBackend be;
    std::string err;
    if (!be.open(0, err)) {
        std::cout << "    \033[1;33m[SKIP]\033[0m open(/dev/npu0) failed: " << err
                  << std::endl;
        return;
    }
    std::cout << "    Opened " << npu_device_node(0) << " (fd=" << be.fd() << ")"
              << std::endl;

    const size_t kBytes = 1u << 20;
    DirectDmaBuffer buf;
    TEST_ASSERT(be.alloc_dma(kBytes, 0, buf, err), "dma-heap allocation failed: " + err);
    TEST_ASSERT(buf.valid() && buf.size >= kBytes, "buffer not usable");
    TEST_ASSERT(buf.vaddr != nullptr, "buffer not mapped");
    std::cout << "    Allocated " << buf.size << " bytes from " << dma_heap_node(0)
              << " (fd=" << buf.fd << ") with no vendor library." << std::endl;

    // Round-trip a pattern through the mapping.
    auto *p = static_cast<volatile uint32_t *>(buf.vaddr);
    const size_t words = buf.size / sizeof(uint32_t);
    for (size_t i = 0; i < words; ++i) p[i] = static_cast<uint32_t>(i * 2654435761u);
    TEST_ASSERT(be.flush_cache(buf), "DMA_BUF_IOCTL_SYNC failed");
    bool ok = true;
    for (size_t i = 0; i < words; ++i) {
        if (p[i] != static_cast<uint32_t>(i * 2654435761u)) { ok = false; break; }
    }
    TEST_ASSERT(ok, "dma-heap mapping did not round-trip");
    std::cout << "    Wrote, synced and read back " << words << " words." << std::endl;

    // Reported, not asserted: see the file header.
    const int prep = be.prepare_dma(buf);
    if (prep == 0) {
        std::cout << "    PREPARE_DMA(fd as u32) accepted by the kernel." << std::endl;
        const int unprep = be.unprepare_dma(buf);
        std::cout << "    UNPREPARE_DMA -> " << unprep << std::endl;
    } else {
        std::cout << "    \033[1;33m[OPEN]\033[0m PREPARE_DMA returned " << prep << " ("
                  << std::strerror(errno) << "): the 4-byte payload hypothesis is wrong"
                     "\n          or the fd must be imported first. Documented in"
                     "\n          docs/DIRECT_KERNEL_IOCTL_SPEC.md." << std::endl;
    }

    buf.release();
    TEST_ASSERT(!buf.valid(), "buffer still valid after release");
    be.close();
    std::cout << "    Released cleanly." << std::endl;
}

} // namespace

int main() {
    std::cout << "========================================================\n"
              << "  ESWIN Direct Kernel Mode Test Suite (Milestone 11)    \n"
              << "========================================================\n\n";

    test_ioctl_encoding_table();
    test_node_naming();
    test_graceful_absence();
    test_on_hardware();

    std::cout << "\n========================================================\n"
              << "  ALL MILESTONE 11 DIRECT KERNEL TESTS PASSED!          \n"
              << "========================================================\n";
    std::cout << "Note: model loading (SUBMIT_MODEL) still needs npu_model_desc reverse\n"
                 "engineering; this suite covers allocation, mapping and SMMU plumbing.\n";
    return 0;
}
