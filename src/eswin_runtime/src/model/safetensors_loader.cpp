/**
 * @file safetensors_loader.cpp
 * @brief Zero-dependency .safetensors reader (Milestone 8).
 *
 * See safetensors_loader.hpp for the container layout. The validation order here is
 * deliberate: the header length is checked against the file size before the JSON is
 * parsed, and each tensor's byte range and element count are checked against the data
 * section before the tensor is published, so a truncated download or a hand-edited
 * header fails with a message naming the offending tensor instead of faulting later
 * during a gather.
 */

#include "safetensors_loader.hpp"
#include "simple_json.hpp"
#include "embedding_table.hpp"   // es_f16_to_f32 / es_bf16_to_f32

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <sstream>

namespace eswin {

SafetensorsLoader::~SafetensorsLoader() {
    close();
}

SafetensorsLoader::SafetensorsLoader(SafetensorsLoader &&other) noexcept
    : m_base(other.m_base)
    , m_map_len(other.m_map_len)
    , m_data_offset(other.m_data_offset)
    , m_fd(other.m_fd)
    , m_open_ms(other.m_open_ms)
    , m_tensors(std::move(other.m_tensors))
    , m_metadata(std::move(other.m_metadata)) {
    other.m_base = nullptr;
    other.m_map_len = 0;
    other.m_data_offset = 0;
    other.m_fd = -1;
    other.m_open_ms = 0.0;
}

SafetensorsLoader &SafetensorsLoader::operator=(SafetensorsLoader &&other) noexcept {
    if (this != &other) {
        close();
        m_base = other.m_base;
        m_map_len = other.m_map_len;
        m_data_offset = other.m_data_offset;
        m_fd = other.m_fd;
        m_open_ms = other.m_open_ms;
        m_tensors = std::move(other.m_tensors);
        m_metadata = std::move(other.m_metadata);
        other.m_base = nullptr;
        other.m_map_len = 0;
        other.m_data_offset = 0;
        other.m_fd = -1;
        other.m_open_ms = 0.0;
    }
    return *this;
}

void SafetensorsLoader::close() {
    if (m_base && m_base != MAP_FAILED) {
        ::munmap(m_base, m_map_len);
    }
    if (m_fd >= 0) {
        ::close(m_fd);
    }
    m_base = nullptr;
    m_map_len = 0;
    m_data_offset = 0;
    m_fd = -1;
    m_open_ms = 0.0;
    m_tensors.clear();
    m_metadata.clear();
}

size_t SafetensorsLoader::dtype_size(const std::string &dtype) {
    if (dtype == "BOOL" || dtype == "U8" || dtype == "I8") return 1;
    if (dtype == "F16" || dtype == "BF16" || dtype == "I16" || dtype == "U16") return 2;
    if (dtype == "F32" || dtype == "I32" || dtype == "U32") return 4;
    if (dtype == "F64" || dtype == "I64" || dtype == "U64") return 8;
    return 0;
}

size_t SafetensorsLoader::element_count(const SafetensorInfo &t) {
    size_t n = 1;
    for (int64_t d : t.shape) {
        if (d < 0) return 0;
        n *= static_cast<size_t>(d);
    }
    return t.shape.empty() ? 0 : n;
}

bool SafetensorsLoader::open(const std::string &path, std::string &err) {
    close();

    // JsonValue::parse only writes err on failure, so a caller reusing the same string
    // across loads would otherwise see a stale message attributed to this file.
    err.clear();

    if (path.empty()) { err = "SafetensorsLoader: empty path"; return false; }

    const auto t0 = std::chrono::steady_clock::now();

    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        err = "SafetensorsLoader: cannot open '" + path + "': " + std::strerror(errno);
        return false;
    }

    struct stat st {};
    if (::fstat(fd, &st) != 0) {
        err = "SafetensorsLoader: fstat failed: " + std::string(std::strerror(errno));
        ::close(fd);
        return false;
    }
    const size_t file_len = static_cast<size_t>(st.st_size);
    if (file_len < 8) {
        err = "SafetensorsLoader: file too small to hold a header length (" +
              std::to_string(file_len) + " bytes)";
        ::close(fd);
        return false;
    }

    void *base = ::mmap(nullptr, file_len, PROT_READ, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
        err = "SafetensorsLoader: mmap failed: " + std::string(std::strerror(errno));
        ::close(fd);
        return false;
    }
    ::madvise(base, file_len, MADV_WILLNEED);

    // Little-endian u64 header length, assembled byte-wise so alignment is not assumed.
    const uint8_t *raw = static_cast<const uint8_t *>(base);
    uint64_t header_len = 0;
    for (int i = 7; i >= 0; --i) {
        header_len = (header_len << 8) | raw[i];
    }
    if (header_len == 0 || 8 + header_len > file_len) {
        std::ostringstream os;
        os << "SafetensorsLoader: header length " << header_len
           << " does not fit in a " << file_len << "-byte file (truncated?)";
        err = os.str();
        ::munmap(base, file_len);
        ::close(fd);
        return false;
    }

    std::string header_json(reinterpret_cast<const char *>(raw + 8),
                            static_cast<size_t>(header_len));
    JsonValue root = JsonValue::parse(header_json, err);
    if (!err.empty() || !root.is_object()) {
        err = "SafetensorsLoader: bad JSON header: " + (err.empty() ? std::string("not an object") : err);
        ::munmap(base, file_len);
        ::close(fd);
        return false;
    }

    const size_t data_offset = static_cast<size_t>(8 + header_len);
    const size_t data_len = file_len - data_offset;

    for (const auto &kv : root.as_object()) {
        if (kv.first == "__metadata__") {
            if (kv.second.is_object()) {
                for (const auto &m : kv.second.as_object()) {
                    m_metadata[m.first] = m.second.as_string();
                }
            }
            continue;
        }
        if (!kv.second.is_object()) {
            err = "SafetensorsLoader: entry '" + kv.first + "' is not an object";
            ::munmap(base, file_len);
            ::close(fd);
            return false;
        }

        SafetensorInfo info;
        info.name = kv.first;
        info.dtype = kv.second["dtype"].as_string();
        for (const auto &d : kv.second["shape"].as_array()) {
            info.shape.push_back(d.as_int64(0));
        }

        const auto &offs = kv.second["data_offsets"].as_array();
        if (offs.size() != 2) {
            err = "SafetensorsLoader: tensor '" + kv.first + "' has no data_offsets pair";
            ::munmap(base, file_len);
            ::close(fd);
            return false;
        }
        const int64_t begin = offs[0].as_int64(-1);
        const int64_t end = offs[1].as_int64(-1);
        if (begin < 0 || end < begin) {
            err = "SafetensorsLoader: tensor '" + kv.first + "' has an invalid byte range";
            ::munmap(base, file_len);
            ::close(fd);
            return false;
        }
        info.offset = static_cast<size_t>(begin);
        info.length = static_cast<size_t>(end - begin);

        const size_t elem = dtype_size(info.dtype);
        if (elem == 0) {
            err = "SafetensorsLoader: tensor '" + kv.first + "' has unsupported dtype '" +
                  info.dtype + "'";
            ::munmap(base, file_len);
            ::close(fd);
            return false;
        }
        if (info.offset + info.length > data_len) {
            std::ostringstream os;
            os << "SafetensorsLoader: tensor '" << kv.first << "' spans bytes " << info.offset
               << ".." << (info.offset + info.length) << " but the data section is only "
               << data_len << " bytes (file truncated?)";
            err = os.str();
            ::munmap(base, file_len);
            ::close(fd);
            return false;
        }
        const size_t expect = element_count(info) * elem;
        if (expect != info.length) {
            std::ostringstream os;
            os << "SafetensorsLoader: tensor '" << kv.first << "' declares " << info.length
               << " bytes but its shape x dtype implies " << expect;
            err = os.str();
            ::munmap(base, file_len);
            ::close(fd);
            return false;
        }

        m_tensors[info.name] = info;
    }

    m_base = base;
    m_map_len = file_len;
    m_data_offset = data_offset;
    m_fd = fd;

    const auto t1 = std::chrono::steady_clock::now();
    m_open_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return true;
}

std::vector<std::string> SafetensorsLoader::tensor_names() const {
    std::vector<std::string> out;
    out.reserve(m_tensors.size());
    for (const auto &kv : m_tensors) out.push_back(kv.first);
    return out;
}

const SafetensorInfo *SafetensorsLoader::find(const std::string &name) const {
    auto it = m_tensors.find(name);
    return (it == m_tensors.end()) ? nullptr : &it->second;
}

const void *SafetensorsLoader::data(const std::string &name, size_t *out_bytes) const {
    const SafetensorInfo *info = find(name);
    if (!info || !m_base) return nullptr;
    if (out_bytes) *out_bytes = info->length;
    return static_cast<const char *>(m_base) + m_data_offset + info->offset;
}

bool SafetensorsLoader::to_f32(const std::string &name, std::vector<float> &out,
                               std::string &err) const {
    const SafetensorInfo *info = find(name);
    if (!info) {
        err = "SafetensorsLoader: no tensor named '" + name + "'";
        return false;
    }
    size_t bytes = 0;
    const void *src = data(name, &bytes);
    if (!src) {
        err = "SafetensorsLoader: tensor '" + name + "' is not mapped";
        return false;
    }

    const size_t n = element_count(*info);
    out.assign(n, 0.0f);

    if (info->dtype == "F32") {
        std::memcpy(out.data(), src, n * sizeof(float));
        return true;
    }
    if (info->dtype == "F64") {
        const double *d = static_cast<const double *>(src);
        for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(d[i]);
        return true;
    }
    if (info->dtype == "F16") {
        const uint16_t *h = static_cast<const uint16_t *>(src);
        for (size_t i = 0; i < n; ++i) out[i] = es_f16_to_f32(h[i]);
        return true;
    }
    if (info->dtype == "BF16") {
        const uint16_t *b = static_cast<const uint16_t *>(src);
        for (size_t i = 0; i < n; ++i) out[i] = es_bf16_to_f32(b[i]);
        return true;
    }
    if (info->dtype == "I64") {
        const int64_t *v = static_cast<const int64_t *>(src);
        for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(v[i]);
        return true;
    }
    if (info->dtype == "I32") {
        const int32_t *v = static_cast<const int32_t *>(src);
        for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(v[i]);
        return true;
    }
    if (info->dtype == "I16") {
        const int16_t *v = static_cast<const int16_t *>(src);
        for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(v[i]);
        return true;
    }
    if (info->dtype == "I8") {
        const int8_t *v = static_cast<const int8_t *>(src);
        for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(v[i]);
        return true;
    }
    if (info->dtype == "U8") {
        const uint8_t *v = static_cast<const uint8_t *>(src);
        for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(v[i]);
        return true;
    }
    if (info->dtype == "BOOL") {
        const uint8_t *v = static_cast<const uint8_t *>(src);
        for (size_t i = 0; i < n; ++i) out[i] = v[i] ? 1.0f : 0.0f;
        return true;
    }

    err = "SafetensorsLoader: unsupported dtype '" + info->dtype + "' for tensor '" + name + "'";
    return false;
}

} // namespace eswin
