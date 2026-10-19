#ifndef ESWIN_SHA256_HPP
#define ESWIN_SHA256_HPP

#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>

namespace eswin {

class Sha256 {
public:
    Sha256();
    void update(const uint8_t *data, size_t length);
    void update(const std::string &str);
    std::string finalize();

    // Convenience static helpers
    static std::string hash_string(const std::string &str);
    static std::string hash_file(const std::string &filepath);

private:
    void transform(const uint8_t chunk[64]);

    uint32_t m_state[8];
    uint64_t m_bitlen;
    uint8_t  m_buffer[64];
    size_t   m_buflen;
};

} // namespace eswin

#endif // ESWIN_SHA256_HPP
