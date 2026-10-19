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
     */
    bool submit_task(
        uint16_t die_id,
        uint32_t model_id,
        const std::vector<DmaBuffer*> &inputs,
        const std::vector<DmaBuffer*> &outputs,
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
