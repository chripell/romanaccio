/**
 * @file embedding_table.cpp
 * @brief Zero-copy memory-mapped embedding table (Milestone 8).
 *
 * See embedding_table.hpp for rationale. Two implementation details that matter on
 * this platform:
 *
 *  1. The mapping is MAP_SHARED + PROT_READ. MAP_PRIVATE would also work for reads but
 *     MAP_SHARED lets the page cache be shared with any other process using the same
 *     model package, which matters when the runtime and a benchmark tool both map the
 *     466 MB Qwen2.5 table.
 *  2. MADV_WILLNEED is issued but the mapping is never touched at open() time, so
 *     open() stays O(1) in file size. The pages fault in during prefill, when the
 *     embeddings are actually gathered.
 */

#include "embedding_table.hpp"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <sstream>

namespace eswin {

namespace {

size_t precision_bytes(EmbeddingPrecision p) {
    switch (p) {
        case EmbeddingPrecision::Fp32: return 4;
        case EmbeddingPrecision::Fp16: return 2;
        case EmbeddingPrecision::Bf16: return 2;
    }
    return 0;
}

} // namespace

float es_f16_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x3FFu;

    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;  // signed zero
        } else {
            // Subnormal: renormalize into the fp32 exponent range.
            int32_t e = -1;
            uint32_t m = mant;
            do {
                ++e;
                m <<= 1;
            } while ((m & 0x400u) == 0);
            bits = sign | (static_cast<uint32_t>(127 - 15 - e) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);  // inf / nan
    } else {
        bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }

    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

float es_bf16_to_f32(uint16_t b) {
    const uint32_t bits = static_cast<uint32_t>(b) << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

EmbeddingTable::~EmbeddingTable() {
    close();
}

EmbeddingTable::EmbeddingTable(EmbeddingTable &&other) noexcept
    : m_base(other.m_base)
    , m_map_len(other.m_map_len)
    , m_rows(other.m_rows)
    , m_cols(other.m_cols)
    , m_elem_bytes(other.m_elem_bytes)
    , m_precision(other.m_precision)
    , m_open_ms(other.m_open_ms)
    , m_fd(other.m_fd) {
    other.m_base = nullptr;
    other.m_map_len = 0;
    other.m_rows = 0;
    other.m_cols = 0;
    other.m_elem_bytes = 0;
    other.m_open_ms = 0.0;
    other.m_fd = -1;
}

EmbeddingTable &EmbeddingTable::operator=(EmbeddingTable &&other) noexcept {
    if (this != &other) {
        close();
        m_base = other.m_base;
        m_map_len = other.m_map_len;
        m_rows = other.m_rows;
        m_cols = other.m_cols;
        m_elem_bytes = other.m_elem_bytes;
        m_precision = other.m_precision;
        m_open_ms = other.m_open_ms;
        m_fd = other.m_fd;
        other.m_base = nullptr;
        other.m_map_len = 0;
        other.m_rows = 0;
        other.m_cols = 0;
        other.m_elem_bytes = 0;
        other.m_open_ms = 0.0;
        other.m_fd = -1;
    }
    return *this;
}

void EmbeddingTable::close() {
    if (m_base && m_base != MAP_FAILED) {
        ::munmap(m_base, m_map_len);
    }
    if (m_fd >= 0) {
        ::close(m_fd);
    }
    m_base = nullptr;
    m_map_len = 0;
    m_rows = 0;
    m_cols = 0;
    m_elem_bytes = 0;
    m_open_ms = 0.0;
    m_fd = -1;
}

bool EmbeddingTable::open(const std::string &path, size_t vocab_size, size_t hidden_size,
                          EmbeddingPrecision precision, std::string &err) {
    close();
    err.clear();  // out-param: never leave a caller's stale message behind on success

    if (path.empty()) { err = "EmbeddingTable: empty path"; return false; }
    if (vocab_size == 0 || hidden_size == 0) {
        err = "EmbeddingTable: vocab_size and hidden_size must be non-zero";
        return false;
    }

    const auto t0 = std::chrono::steady_clock::now();

    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        err = "EmbeddingTable: cannot open '" + path + "': " + std::strerror(errno);
        return false;
    }

    struct stat st {};
    if (::fstat(fd, &st) != 0) {
        err = "EmbeddingTable: fstat failed: " + std::string(std::strerror(errno));
        ::close(fd);
        return false;
    }

    const size_t elem = precision_bytes(precision);
    const size_t expected = vocab_size * hidden_size * elem;
    const size_t actual = static_cast<size_t>(st.st_size);
    if (actual != expected) {
        std::ostringstream os;
        os << "EmbeddingTable: size mismatch for '" << path << "': file is " << actual
           << " bytes but " << vocab_size << " x " << hidden_size << " x " << elem
           << " requires " << expected << " bytes";
        err = os.str();
        ::close(fd);
        return false;
    }
    if (actual == 0) {
        err = "EmbeddingTable: refusing to map an empty file";
        ::close(fd);
        return false;
    }

    void *base = ::mmap(nullptr, actual, PROT_READ, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
        err = "EmbeddingTable: mmap failed: " + std::string(std::strerror(errno));
        ::close(fd);
        return false;
    }

    // Hint readahead without blocking: keeps open() O(1) in file size.
    ::madvise(base, actual, MADV_WILLNEED);

    m_base = base;
    m_map_len = actual;
    m_rows = vocab_size;
    m_cols = hidden_size;
    m_elem_bytes = elem;
    m_precision = precision;
    m_fd = fd;

    const auto t1 = std::chrono::steady_clock::now();
    m_open_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return true;
}

bool EmbeddingTable::open_auto(const std::string &path, size_t vocab_size, size_t hidden_size,
                               std::string &err) {
    struct stat st {};
    if (::stat(path.c_str(), &st) != 0) {
        err = "EmbeddingTable: cannot stat '" + path + "': " + std::strerror(errno);
        return false;
    }
    const size_t actual = static_cast<size_t>(st.st_size);
    const size_t elems = vocab_size * hidden_size;
    if (elems == 0) {
        err = "EmbeddingTable: vocab_size and hidden_size must be non-zero";
        return false;
    }

    if (actual == elems * 4) {
        return open(path, vocab_size, hidden_size, EmbeddingPrecision::Fp32, err);
    }
    if (actual == elems * 2) {
        // 2 bytes per element is ambiguous between fp16 and bf16; the ESWIN packages
        // export fp16, so that is the default and bf16 must be requested explicitly.
        return open(path, vocab_size, hidden_size, EmbeddingPrecision::Fp16, err);
    }

    std::ostringstream os;
    os << "EmbeddingTable: cannot infer precision for '" << path << "': file is " << actual
       << " bytes, but " << vocab_size << " x " << hidden_size << " needs "
       << (elems * 2) << " (fp16/bf16) or " << (elems * 4) << " (fp32)";
    err = os.str();
    return false;
}

const void *EmbeddingTable::raw_row(size_t token) const {
    if (!m_base || token >= m_rows) return nullptr;
    return static_cast<const char *>(m_base) + token * m_cols * m_elem_bytes;
}

bool EmbeddingTable::get_row(size_t token, float *dst, size_t dst_floats) const {
    const void *row = raw_row(token);
    if (!row || !dst || dst_floats < m_cols) return false;

    switch (m_precision) {
        case EmbeddingPrecision::Fp32: {
            std::memcpy(dst, row, m_cols * sizeof(float));
            break;
        }
        case EmbeddingPrecision::Fp16: {
            const uint16_t *src = static_cast<const uint16_t *>(row);
            for (size_t i = 0; i < m_cols; ++i) dst[i] = es_f16_to_f32(src[i]);
            break;
        }
        case EmbeddingPrecision::Bf16: {
            const uint16_t *src = static_cast<const uint16_t *>(row);
            for (size_t i = 0; i < m_cols; ++i) dst[i] = es_bf16_to_f32(src[i]);
            break;
        }
    }
    return true;
}

size_t EmbeddingTable::gather(const int32_t *tokens, size_t n, float *dst,
                              size_t dst_floats) const {
    if (!m_base || !tokens || !dst) return 0;
    size_t written = 0;
    for (size_t i = 0; i < n; ++i) {
        float *out = dst + i * m_cols;
        if ((i + 1) * m_cols > dst_floats) break;
        if (tokens[i] < 0 || static_cast<size_t>(tokens[i]) >= m_rows ||
            !get_row(static_cast<size_t>(tokens[i]), out, m_cols)) {
            // Out-of-range token: emit zeros rather than reading past the mapping.
            std::memset(out, 0, m_cols * sizeof(float));
        }
        ++written;
    }
    return written;
}

void EmbeddingTable::advise_willneed() const {
    if (m_base && m_map_len) {
        ::madvise(m_base, m_map_len, MADV_WILLNEED);
    }
}

} // namespace eswin
