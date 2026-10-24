/**
 * @file safetensors_loader.hpp
 * @brief Zero-dependency Hugging Face .safetensors reader (Milestone 8).
 *
 * Container layout:
 *   [0..8)                     little-endian u64 header length N
 *   [8..8+N)                   JSON header: name -> {dtype, shape, data_offsets}
 *   [8+N..EOF)                 densely packed tensor data
 *
 * The whole file is mapped read-only, so tensor access is a pointer into the mapping
 * with no copy. Only the JSON header is copied out for parsing (a few hundred KB at
 * most, versus gigabytes of weights).
 *
 * Every data_offsets range is validated against the declared shape, dtype element size
 * and the mapped data-section length before a tensor is exposed, so a truncated or
 * hand-edited file is rejected instead of read out of bounds.
 *
 * Note: tensor order comes from the JSON object map and is therefore alphabetical, not
 * file order. Use find()/data() by name rather than relying on iteration order.
 */

#ifndef ES_SAFETENSORS_LOADER_HPP
#define ES_SAFETENSORS_LOADER_HPP

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace eswin {

struct SafetensorInfo {
    std::string          name;
    std::string          dtype;    ///< "F32", "F16", "BF16", "I64", "I32", "I8", "U8", "BOOL", ...
    std::vector<int64_t> shape;
    size_t               offset = 0;  ///< Byte offset within the data section.
    size_t               length = 0;  ///< Byte length of the tensor payload.
};

class SafetensorsLoader {
public:
    SafetensorsLoader() = default;
    ~SafetensorsLoader();

    SafetensorsLoader(SafetensorsLoader &&other) noexcept;
    SafetensorsLoader &operator=(SafetensorsLoader &&other) noexcept;
    SafetensorsLoader(const SafetensorsLoader &) = delete;
    SafetensorsLoader &operator=(const SafetensorsLoader &) = delete;

    /// Maps and validates a .safetensors file. @p err receives a specific reason.
    bool open(const std::string &path, std::string &err);
    void close();

    bool   is_open() const noexcept { return m_base != nullptr; }
    size_t tensor_count() const noexcept { return m_tensors.size(); }
    double open_ms() const noexcept { return m_open_ms; }
    size_t mapped_bytes() const noexcept { return m_map_len; }

    /// Tensor names, alphabetical.
    std::vector<std::string> tensor_names() const;

    /// Metadata block ("__metadata__"), empty when absent.
    const std::map<std::string, std::string> &metadata() const noexcept { return m_metadata; }

    const SafetensorInfo *find(const std::string &name) const;

    /// Pointer into the mapping for @p name, or nullptr. Optionally returns byte length.
    const void *data(const std::string &name, size_t *out_bytes = nullptr) const;

    /// Product of the shape entries (1 for a scalar).
    static size_t element_count(const SafetensorInfo &t);

    /// Bytes per element for a safetensors dtype name; 0 when unsupported.
    static size_t dtype_size(const std::string &dtype);

    /**
     * @brief Converts one tensor to fp32 regardless of its stored dtype.
     *
     * Supports F64/F32/F16/BF16/I64/I32/I16/I8/U8/BOOL. Fails with a specific message
     * for an unknown dtype rather than reinterpreting the bytes.
     */
    bool to_f32(const std::string &name, std::vector<float> &out, std::string &err) const;

private:
    void *m_base = nullptr;
    size_t m_map_len = 0;
    size_t m_data_offset = 0;      ///< Where the tensor data section begins in the file.
    int    m_fd = -1;
    double m_open_ms = 0.0;
    std::map<std::string, SafetensorInfo> m_tensors;
    std::map<std::string, std::string>    m_metadata;
};

} // namespace eswin

#endif // ES_SAFETENSORS_LOADER_HPP
