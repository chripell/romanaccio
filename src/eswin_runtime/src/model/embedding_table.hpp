/**
 * @file embedding_table.hpp
 * @brief Zero-copy memory-mapped embedding table (Milestone 8).
 *
 * LLM embedding tables are large relative to everything else the runtime touches at
 * startup: 466 MB fp16 for Qwen2.5-Coder-1.5B (151936 x 1536), 1.08 GB for
 * DeepSeek-7B. Reading them into a std::vector costs a full copy plus the heap
 * footprint, which shows up directly as cold-start latency on the board.
 *
 * This maps the file read-only with MAP_SHARED and MADV_WILLNEED instead, so:
 *   - open() cost is independent of file size (page tables only, pages fault in on use);
 *   - no second copy of the table ever exists in RAM;
 *   - the kernel page cache is shared with any other process using the same package.
 *
 * Rows are converted to fp32 on lookup. fp16 and bf16 are both 2 bytes per element,
 * so precision cannot be inferred from file size alone; the caller states it (default
 * fp16, which is what the ESWIN packages ship) and the loader validates the size
 * against vocab_size x hidden_size x element size.
 */

#ifndef ES_EMBEDDING_TABLE_HPP
#define ES_EMBEDDING_TABLE_HPP

#include <cstddef>
#include <cstdint>
#include <string>

namespace eswin {

/// Element encoding of a table. These are identifiers, not byte sizes: fp16 and bf16
/// are both 2 bytes, which is exactly why precision cannot be inferred from size alone.
enum class EmbeddingPrecision {
    Fp32,
    Fp16,
    Bf16,
};

/// Read-only, move-only view over a memory-mapped embedding table.
class EmbeddingTable {
public:
    EmbeddingTable() = default;
    ~EmbeddingTable();

    EmbeddingTable(EmbeddingTable &&other) noexcept;
    EmbeddingTable &operator=(EmbeddingTable &&other) noexcept;
    EmbeddingTable(const EmbeddingTable &) = delete;
    EmbeddingTable &operator=(const EmbeddingTable &) = delete;

    /**
     * @brief Maps an embedding table and validates its size against the declared shape.
     *
     * @param path        Table file (e.g. "embedding.bin").
     * @param vocab_size  Expected row count.
     * @param hidden_size Expected column count.
     * @param precision   Element encoding; must match how the file was exported.
     * @param err         Populated with a specific reason on failure.
     */
    bool open(const std::string &path, size_t vocab_size, size_t hidden_size,
              EmbeddingPrecision precision, std::string &err);

    /**
     * @brief Maps a table inferring fp32 vs 2-byte encoding from the file size.
     *
     * A 2-byte file is treated as fp16 (the ESWIN convention); pass an explicit
     * precision to open() for bf16. Fails with a specific error when the size matches
     * neither encoding, rather than silently reading garbage.
     */
    bool open_auto(const std::string &path, size_t vocab_size, size_t hidden_size,
                   std::string &err);

    void close();

    bool is_open() const noexcept { return m_base != nullptr; }
    size_t rows() const noexcept { return m_rows; }
    size_t cols() const noexcept { return m_cols; }
    EmbeddingPrecision precision() const noexcept { return m_precision; }
    size_t mapped_bytes() const noexcept { return m_map_len; }
    /// Wall time of the most recent successful open(), for cold-start assertions.
    double open_ms() const noexcept { return m_open_ms; }

    /**
     * @brief Copies one token's embedding into @p dst as fp32.
     * @return false if the token is out of range or dst cannot hold cols() floats.
     */
    bool get_row(size_t token, float *dst, size_t dst_floats) const;

    /**
     * @brief Gathers @p n token rows contiguously into @p dst (n * cols() floats).
     * @return Number of rows actually written; out-of-range tokens are zero-filled.
     */
    size_t gather(const int32_t *tokens, size_t n, float *dst, size_t dst_floats) const;

    /// Raw (unconverted) row pointer, or nullptr when out of range.
    const void *raw_row(size_t token) const;

    /// Prefault the mapping; only useful when measuring steady-state decode.
    void advise_willneed() const;

private:
    void *m_base = nullptr;
    size_t m_map_len = 0;
    size_t m_rows = 0;
    size_t m_cols = 0;
    size_t m_elem_bytes = 0;
    EmbeddingPrecision m_precision = EmbeddingPrecision::Fp16;
    double m_open_ms = 0.0;
    int m_fd = -1;
};

/// Portable IEEE-754 half to float. Matches the RISC-V _Float16 path bit-for-bit for
/// finite values; kept here so model/ does not depend on llm/.
float es_f16_to_f32(uint16_t h);
/// bfloat16 is the truncated top half of an fp32, so conversion is a shift.
float es_bf16_to_f32(uint16_t b);

} // namespace eswin

#endif // ES_EMBEDDING_TABLE_HPP
