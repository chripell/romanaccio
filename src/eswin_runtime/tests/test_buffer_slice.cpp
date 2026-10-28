/**
 * @file test_buffer_slice.cpp
 * @brief Milestone 13 test suite: slice marshalling for expert dispatch.
 *
 * WorkerPool::submit_task() cannot be exercised off-board -- host-fallback simulates and
 * never marshals a descriptor -- so the marshalling lives in the pure helper
 * fill_ipc_buffers() and is tested here directly. What the worker does with a descriptor is
 * already silicon-proven (FINDINGS #26/#27/#30 bound experts through DevBuf.offset).
 *
 * Two invariants matter most:
 *   - the whole-buffer default reproduces exactly what the pre-slice code put on the wire
 *     (offset 0, size = buffer->size()), so every silicon-validated caller is unaffected;
 *   - every request that would make the NPU read outside a prepared MMZ region is rejected
 *     rather than clamped, because a silent clamp is the Issue #47 corruption class.
 */

#include "../src/backend/worker_pool.hpp"

#include <cstdlib>
#include <iostream>
#include <vector>

#define TEST_ASSERT(cond, msg)                                                  \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::cerr << "\033[1;31m[FAIL]\033[0m " << msg << " (" << #cond     \
                      << ") at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

#define TEST_OK(expr, msg)                                                      \
    do {                                                                        \
        SliceError e_ = (expr);                                                 \
        if (e_ != SliceError::Success) {                                        \
            std::cerr << "\033[1;31m[FAIL]\033[0m " << msg << " -> "             \
                      << describe_slice_error(e_) << " at " << __FILE__ << ":"    \
                      << __LINE__ << std::endl;                                 \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

using namespace eswin;

namespace {

int g_pass = 0;

void ok(const char *label) {
    std::cout << "\033[1;32m[PASS]\033[0m " << label << std::endl;
    ++g_pass;
}

constexpr size_t kBlobBytes = 512 * 1024;      // stands in for a resident expert blob
constexpr size_t kExpertBytes = kBlobBytes / 2;

SliceError marshal(const std::vector<DmaBufferSlice> &slices, IpcBufferDesc *out, size_t &n) {
    return fill_ipc_buffers(slices, out, kIpcBufferSlots, n);
}

// --- Whole-buffer compatibility ------------------------------------------------

void test_default_slice_reproduces_legacy_wire_content() {
    DmaBuffer buf(kBlobBytes);
    TEST_ASSERT(buf.is_valid(), "host-fallback allocation failed");

    IpcBufferDesc d[kIpcBufferSlots]{};
    size_t n = 0;
    TEST_OK(marshal({DmaBufferSlice{&buf}}, d, n), "whole-buffer default rejected");
    TEST_ASSERT(n == 1, "one slice must marshal one descriptor");
    // Exactly the values the pre-Milestone-13 code hardcoded.
    TEST_ASSERT(d[0].mem_fd == buf.mem_fd(), "mem_fd not propagated");
    TEST_ASSERT(d[0].offset == 0, "whole-buffer offset must be 0");
    TEST_ASSERT(d[0].size == buf.size(), "whole-buffer size must be the full allocation");
    TEST_ASSERT(d[0].reserved == 0, "reserved must stay zero");
    ok("whole-buffer slice reproduces legacy descriptor");
}

// --- Expert selection ---------------------------------------------------------

void test_expert_slice_propagates_offset_and_length() {
    DmaBuffer blob(kBlobBytes);
    IpcBufferDesc d[kIpcBufferSlots]{};
    size_t n = 0;
    TEST_OK(marshal({DmaBufferSlice{&blob, kExpertBytes, kExpertBytes}}, d, n),
            "second-expert slice rejected");
    TEST_ASSERT(d[0].offset == kExpertBytes, "expert offset dropped");
    TEST_ASSERT(d[0].size == kExpertBytes, "size must be the SLICE length, not the allocation");
    ok("expert slice carries {mem_fd, offset, length}");
}

void test_two_experts_from_one_blob_are_distinct() {
    DmaBuffer blob(kBlobBytes);
    IpcBufferDesc d[kIpcBufferSlots]{};
    size_t n = 0;
    TEST_OK(marshal({DmaBufferSlice{&blob, 0, kExpertBytes},
                     DmaBufferSlice{&blob, kExpertBytes, kExpertBytes}}, d, n),
            "two-expert dispatch rejected");
    TEST_ASSERT(n == 2, "both slots must be written");
    TEST_ASSERT(d[0].mem_fd == d[1].mem_fd, "one blob must be prepared once, as one mem_fd");
    TEST_ASSERT(d[0].offset != d[1].offset, "same offset for two experts selects the same weights");
    TEST_ASSERT(d[0].size == d[1].size, "experts in one blob are equal-sized");
    ok("two experts bind through two slices of one buffer");
}

void test_zero_size_means_rest_of_buffer() {
    DmaBuffer blob(kBlobBytes);
    IpcBufferDesc d[kIpcBufferSlots]{};
    size_t n = 0;
    TEST_OK(marshal({DmaBufferSlice{&blob, kExpertBytes, 0}}, d, n), "size=0 slice rejected");
    TEST_ASSERT(d[0].size == kBlobBytes - kExpertBytes, "size=0 must mean rest-of-buffer");
    ok("size 0 resolves to the remaining bytes");
}

void test_exact_fit_at_buffer_end_is_allowed() {
    DmaBuffer blob(kBlobBytes);
    IpcBufferDesc d[kIpcBufferSlots]{};
    size_t n = 0;
    TEST_OK(marshal({DmaBufferSlice{&blob, kBlobBytes - kExpertBytes, kExpertBytes}}, d, n),
            "slice ending exactly at the buffer end must be valid");
    TEST_ASSERT(d[0].offset + d[0].size == kBlobBytes, "descriptor must cover to the last byte");
    ok("boundary-exact slice accepted");
}

// --- Rejections ---------------------------------------------------------------

void test_rejects_null_buffer() {
    IpcBufferDesc d[kIpcBufferSlots]{};
    size_t n = 0;
    TEST_ASSERT(marshal({DmaBufferSlice{nullptr}}, d, n) == SliceError::NullBuffer,
                "null buffer must be rejected");
    ok("null buffer rejected");
}

void test_reports_index_of_offending_slice() {
    DmaBuffer blob(kBlobBytes);
    IpcBufferDesc d[kIpcBufferSlots]{};
    size_t n = 0;
    auto err = marshal({DmaBufferSlice{&blob, 0, kExpertBytes},
                        DmaBufferSlice{nullptr},
                        DmaBufferSlice{&blob, kExpertBytes, kExpertBytes}}, d, n);
    TEST_ASSERT(err == SliceError::NullBuffer, "second slice is the null one");
    TEST_ASSERT(n == 1, "out_count must be the failing slot index, not a total");
    ok("failing slot index reported");
}

void test_rejects_unaligned_offset() {
    DmaBuffer blob(kBlobBytes);
    IpcBufferDesc d[kIpcBufferSlots]{};
    size_t n = 0;
    // Sub-page offsets are the untested case: refuse instead of guessing.
    TEST_ASSERT(marshal({DmaBufferSlice{&blob, 100, kExpertBytes}}, d, n) == SliceError::UnalignedOffset,
                "offset 100 must be rejected");
    TEST_ASSERT(marshal({DmaBufferSlice{&blob, kExpertBytes + 1, kExpertBytes}}, d, n) ==
                    SliceError::UnalignedOffset,
                "offset 4097 must be rejected");
    TEST_ASSERT(n == 0, "nothing may be written before the rejection");
    ok("non-page-aligned offset rejected");
}

void test_rejects_range_past_buffer() {
    DmaBuffer blob(kBlobBytes);
    IpcBufferDesc d[kIpcBufferSlots]{};
    size_t n = 0;
    TEST_ASSERT(marshal({DmaBufferSlice{&blob, kBlobBytes, kExpertBytes}}, d, n) ==
                    SliceError::RangeExceedsBuffer,
                "offset past the end must be rejected");
    TEST_ASSERT(marshal({DmaBufferSlice{&blob, kExpertBytes, kBlobBytes}}, d, n) ==
                    SliceError::RangeExceedsBuffer,
                "length past the end must be rejected");
    // Would wrap to a small offset+size if the check were written as an addition.
    TEST_ASSERT(marshal({DmaBufferSlice{&blob, kExpertBytes, SIZE_MAX}}, d, n) ==
                    SliceError::RangeExceedsBuffer,
                " SIZE_MAX length must not wrap the bounds check");
    ok("out-of-range slice rejected (no wraparound)");
}

void test_rejects_empty_slice() {
    DmaBuffer blob(kBlobBytes);
    IpcBufferDesc d[kIpcBufferSlots]{};
    size_t n = 0;
    TEST_ASSERT(marshal({DmaBufferSlice{&blob, kBlobBytes, 0}}, d, n) == SliceError::EmptySlice,
                "a zero-length descriptor is meaningless to the driver");
    TEST_ASSERT(marshal({DmaBufferSlice{&blob, 0, 0}}, d, n) == SliceError::Success,
                "size=0 at offset 0 means the whole buffer, which is not empty");
    ok("zero-length slice rejected");
}

void test_rejects_more_slices_than_slots() {
    DmaBuffer blob(kBlobBytes);
    IpcBufferDesc d[kIpcBufferSlots]{};
    size_t n = 0;
    std::vector<DmaBufferSlice> many(kIpcBufferSlots + 1, DmaBufferSlice{&blob, 0, kSliceAlignment});
    TEST_ASSERT(marshal(many, d, n) == SliceError::TooManyBuffers,
                "over-capacity must be rejected, not truncated");
    ok("slice count above the driver slot limit rejected");
}

// --- WorkerPool surface -------------------------------------------------------

void test_pointer_overload_still_resolves_and_pool_gate() {
    DmaBuffer buf(kBlobBytes);
    WorkerPool pool;  // never started: no channels, so both overloads must fail closed

    std::vector<DmaBuffer*> in_ptrs{&buf}, out_ptrs{&buf};
    TEST_ASSERT(!pool.submit_task(0, 7, in_ptrs, out_ptrs),
                "unstarted pool must not report success");

    std::vector<DmaBufferSlice> in_slices{DmaBufferSlice{&buf}}, out_slices{DmaBufferSlice{&buf}};
    TEST_ASSERT(!pool.submit_task(0, 7, in_slices, out_slices),
                "slice overload must honour the same gate");

    // The MoE call site shape: brace-init of raw pointers must not become ambiguous with
    // the new overload.
    TEST_ASSERT(!pool.submit_task(0, 7, { &buf }, { &buf }), "brace-init call misresolved");
    ok("both overloads compile-side-by-side and fail closed without workers");
}

} // namespace

int main() {
    std::cout << "\033[1;34m=== Milestone 13: BufferSlice marshalling for expert dispatch ===\033[0m"
              << std::endl;

    test_default_slice_reproduces_legacy_wire_content();
    test_expert_slice_propagates_offset_and_length();
    test_two_experts_from_one_blob_are_distinct();
    test_zero_size_means_rest_of_buffer();
    test_exact_fit_at_buffer_end_is_allowed();
    test_rejects_null_buffer();
    test_reports_index_of_offending_slice();
    test_rejects_unaligned_offset();
    test_rejects_range_past_buffer();
    test_rejects_empty_slice();
    test_rejects_more_slices_than_slots();
    test_pointer_overload_still_resolves_and_pool_gate();

    std::cout << "\033[1;32mAll " << g_pass << " buffer-slice tests passed.\033[0m" << std::endl;
    return 0;
}
