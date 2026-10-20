/**
 * @file dsp_driver.hpp
 * @brief Dynamic loader and hardware interface for ESWIN EIC7700X Cadence Vision Q7 DSP cores.
 *
 * Provides a clean C++ abstraction over libes_dsp_runtime_ll.so and libes_memory.so,
 * enabling direct execution of the 372 pre-compiled firmware DSP kernels
 * (/usr/lib/firmware/dsp_kernels/*.pkg.lib) across all 8 DSP cores (/dev/es-dsp0..7).
 */

#ifndef ES_DSP_DRIVER_HPP
#define ES_DSP_DRIVER_HPP

#include <cstdint>
#include <cstddef>
#include <string>
#include <memory>
#include <vector>
#include <functional>
#include <stdexcept>

namespace eswin {

// Core identifiers (0..3 on Die 0, 4..7 on Die 1)
enum class DspCore : int {
    Core0 = 0, Core1 = 1, Core2 = 2, Core3 = 3,
    Core4 = 4, Core5 = 5, Core6 = 6, Core7 = 7
};

// Memory Buffer Descriptor
struct DspBuffer {
    uint64_t mem_fd{0};
    uint64_t offset{0};
    uint64_t size{0};
    uint64_t reserved{0};
    void* host_ptr{nullptr};
    size_t alloc_size{0};
};

/**
 * Low-level DSP context wrapping /dev/es-dspX hardware operations.
 */
class DspContext {
public:
    explicit DspContext(DspCore core = DspCore::Core0);
    ~DspContext();

    DspContext(const DspContext&) = delete;
    DspContext& operator=(const DspContext&) = delete;

    void open();
    void close();
    bool is_open() const { return dsp_fd_ >= 0; }
    DspCore core() const { return core_; }
    int fd() const { return dsp_fd_; }

    // Operator Lifecycle
    uint32_t load_operator(const std::string& op_name, const std::string& op_dir = "/usr/lib/firmware/dsp_kernels");
    void unload_operator(uint32_t handle);

    // Memory Allocation & DMA Mapping
    DspBuffer allocate_buffer(size_t size);
    void free_buffer(DspBuffer& buf);
    void prepare_dma(const DspBuffer& buf);
    void unprepare_dma(const DspBuffer& buf);

    // Task Execution
    void submit_task(
        uint32_t op_handle,
        const std::vector<DspBuffer>& cfgs,
        const std::vector<DspBuffer>& inputs,
        const std::vector<DspBuffer>& outputs,
        bool sync_cache = true
    );

    // Version info
    std::string get_version_string();
    uint64_t get_version();

private:
    DspCore core_;
    int dsp_fd_{-1};
    void* lib_dsp_{nullptr};
    void* lib_mem_{nullptr};

    void load_symbols();
};

} // namespace eswin

#endif // ES_DSP_DRIVER_HPP
