/**
 * @file worker_pool.hpp
 * @brief Multi-Process Dual-Die Isolation & NUMA Worker Daemon for EIC7700X NPUs.
 *
 * Circumvents the vendor driver ES_NPU_ERROR_DEV_MULTI_SET (0xA00F6046) limitation
 * by spawning isolated child worker processes pinned to NUMA node 0 (Die 0) and
 * NUMA node 1 (Die 1) communicating over high-speed bidirectional IPC channels.
 */

#ifndef ES_WORKER_POOL_HPP
#define ES_WORKER_POOL_HPP

#include "driver_wrapper.hpp"
#include "dma_buffer.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <chrono>

namespace eswin {

// =============================================================================
// IPC Protocol Definitions
// =============================================================================

enum class IpcCmd : uint32_t {
    Ping = 1,
    LoadModel,
    UnloadModel,
    PrepareBuffer,
    UnprepareBuffer,
    SubmitTask,
    SubmitExpertBatch,
    SubmitLmHeadBatch,
    Shutdown
};

enum class IpcStatus : int32_t {
    Success = 0,
    ErrorInvalidArg = -1,
    ErrorNotReady = -2,
    ErrorExecutionFailed = -3,
    ErrorShutdown = -4
};

#pragma pack(push, 8)
struct IpcBufferDesc {
    uint64_t mem_fd;
    uint64_t offset;
    uint64_t size;
    uint64_t reserved;
};

struct IpcRequest {
    IpcCmd        cmd;
    uint32_t      seq_id;
    uint32_t      model_id;
    uint8_t       input_count;
    uint8_t       output_count;
    char          model_path[256];
    IpcBufferDesc inputs[10];
    IpcBufferDesc outputs[10];
};

struct IpcResponse {
    IpcStatus status;
    uint32_t  seq_id;
    uint32_t  model_id;
    uint8_t   num_inputs;
    uint8_t   num_outputs;
    uint64_t  input_sizes[10];
    uint64_t  output_sizes[10];
    int32_t   driver_error;
    uint64_t  elapsed_ns;
};
#pragma pack(pop)

struct ModelInfo {
    uint32_t model_id = 0;
    uint8_t  num_inputs = 0;
    uint8_t  num_outputs = 0;
    uint64_t input_sizes[10] = {0};
    uint64_t output_sizes[10] = {0};
};

// =============================================================================
// Buffer Slices (Milestone 13)
// =============================================================================

/**
 * @brief Buffer slots per task: mirrors NpuTask's inputFd[10]/outputFd[10].
 *
 * This is a driver-struct limit, not a convention, so exceeding it is rejected
 * instead of silently truncated (the previous code set input_count from the vector
 * size and then filled only 10 slots, handing the driver uninitialized descriptors).
 */
inline constexpr size_t kIpcBufferSlots = 10;

/**
 * @brief Required alignment of a slice's start inside its buffer.
 *
 * Every offset proven to be honoured by silicon is page-aligned: offset 1,769,472 and
 * 8,908,800 both selected the right expert (FINDINGS #26, #27, #30). Sub-page offsets are
 * untested, so they are refused loudly rather than trusted to quietly read the wrong bytes.
 */
inline constexpr size_t kSliceAlignment = 4096;

/**
 * @brief A window inside a DmaBuffer, submitted as one task tensor without copying.
 *
 * Lets one resident blob serve many models: expert selection for MoE is just a different
 * offset (see MoERouter::expert_blob_offset()), which is what keeps DeepSeek-V2-Lite at 27
 * loadables per die instead of 1,716.
 */
struct DmaBufferSlice {
    const DmaBuffer *buffer = nullptr;
    size_t offset = 0;
    size_t size = 0;  ///< 0 means "rest of the buffer" (buffer->size() - offset).
};

/** @brief Rejection reasons for a slice vector; Success means fully marshalled. */
enum class SliceError : int {
    Success = 0,
    NullBuffer,          ///< A slice has no buffer; the descriptor would be meaningless.
    TooManyBuffers,      ///< More slices than the driver struct has slots.
    UnalignedOffset,     ///< Offset is not a multiple of kSliceAlignment.
    RangeExceedsBuffer,  ///< The slice would read past the end of its buffer.
    EmptySlice,          ///< Zero-length slice.
};

const char *describe_slice_error(SliceError err);

/**
 * @brief Validate a slice vector and write its IPC descriptors. Pure, so it is testable
 *        off-board (host-fallback submit_task simulates and never marshals).
 *
 * @param slices   Input or output slices for one task.
 * @param out      Destination array, at least @p cap entries.
 * @param cap      Slots available in @p out (pass kIpcBufferSlots).
 * @param out_count On Success the number of slices written; on failure the index of the
 *                  offending slice, which is why it is written on every return path.
 */
SliceError fill_ipc_buffers(const std::vector<DmaBufferSlice> &slices,
                            IpcBufferDesc *out, size_t cap, size_t &out_count);

// =============================================================================
// Worker Daemon Interface
// =============================================================================

class WorkerChannel {
public:
    explicit WorkerChannel(uint16_t die_id);
    ~WorkerChannel();

    bool start();
    void shutdown();

    bool send_request(const IpcRequest &req, IpcResponse &resp);

    uint16_t die_id() const noexcept { return m_die_id; }
    pid_t pid() const noexcept { return m_pid; }
    bool is_alive() const noexcept { return m_is_alive; }

private:
    void run_worker_loop();

    uint16_t m_die_id;
    pid_t    m_pid = -1;
    int      m_master_fd = -1;
    int      m_worker_fd = -1;
    bool     m_is_alive = false;
    uint32_t m_seq_counter = 0;
};

class WorkerPool {
public:
    WorkerPool();
    ~WorkerPool();

    // Move-only
    WorkerPool(WorkerPool &&) = default;
    WorkerPool& operator=(WorkerPool &&) = default;
    WorkerPool(const WorkerPool &) = delete;
    WorkerPool& operator=(const WorkerPool &) = delete;

    /**
     * @brief Spawns and initializes worker daemons across detected dies.
     * @param num_dies Number of dies to spawn (defaults to detected hardware dies, min 1, max 2).
     */
    bool start(uint16_t num_dies = 2);

    /**
     * @brief Gracefully terminates worker daemons.
     */
    void shutdown();

    /**
     * @brief Loads a .model file on the target NPU die and returns detailed tensor metadata.
     */
    bool load_model(uint16_t die_id, const std::string &path, ModelInfo &out_info);

    /**
     * @brief Loads a .model file on the target NPU die.
     */
    bool load_model(uint16_t die_id, const std::string &path, uint32_t &out_model_id);

    /**
     * @brief Unloads a model from the target NPU die.
     */
    bool unload_model(uint16_t die_id, uint32_t model_id);

    /**
     * @brief Maps an MMZ DMA buffer into the target NPU die's SMMU page table.
     */
    bool prepare_buffer(uint16_t die_id, const DmaBuffer &buf);

    /**
     * @brief Unmaps an MMZ DMA buffer from the target NPU die's SMMU.
     */
    bool unprepare_buffer(uint16_t die_id, const DmaBuffer &buf);

    /**
     * @brief Dispatches a synchronous task execution to the target NPU die.
     *
     * Whole-buffer convenience form: every tensor is bound at offset 0 for the buffer's
     * full size. It delegates to the slice overload, so existing callers keep exactly the
     * wire content they produced before.
     */
    bool submit_task(
        uint16_t die_id,
        uint32_t model_id,
        const std::vector<DmaBuffer*> &inputs,
        const std::vector<DmaBuffer*> &outputs,
        uint64_t *out_elapsed_ns = nullptr
    );

    /**
     * @brief Dispatches a task whose tensors are windows inside larger buffers.
     *
     * This is the MoE dispatch path: one resident per-die expert blob, and each submit binds
     * the expert the router selected. Preparation stays whole-buffer (the SMMU maps mem_fd,
     * not a range), so a blob is prepared once and sliced many times.
     */
    bool submit_task(
        uint16_t die_id,
        uint32_t model_id,
        const std::vector<DmaBufferSlice> &inputs,
        const std::vector<DmaBufferSlice> &outputs,
        uint64_t *out_elapsed_ns = nullptr
    );

    bool submit_expert_batch(
        uint16_t die_id,
        uint32_t model_id,
        const DmaBufferSlice &act_input,
        const std::vector<DmaBufferSlice> &weight_inputs,
        const std::vector<DmaBufferSlice> &outputs,
        uint64_t *out_elapsed_ns = nullptr
    );

    bool submit_lm_head_batch(
        uint16_t die_id,
        const uint32_t model_ids[4],
        const DmaBufferSlice &act_input,
        const DmaBufferSlice outputs[4],
        uint64_t *out_elapsed_ns = nullptr
    );

    uint16_t active_worker_count() const noexcept {
        return static_cast<uint16_t>(m_channels.size());
    }

private:
    std::vector<std::unique_ptr<WorkerChannel>> m_channels;
    bool m_running = false;
};

} // namespace eswin

#endif // ES_WORKER_POOL_HPP
