#ifndef ESWIN_KV_CACHE_HPP
#define ESWIN_KV_CACHE_HPP

#include <vector>
#include <cstdint>
#include <cstddef>

namespace eswin {

struct KvCacheConfig {
    uint32_t num_layers = 30;
    uint32_t max_seq_len = 2048;
    uint32_t num_kv_heads = 3;
    uint32_t head_dim = 64;
};

class KvCache {
public:
    explicit KvCache(const KvCacheConfig &cfg);

    void reset();

    // Size of single key/value slice across all heads: num_kv_heads * head_dim
    size_t kv_dim() const { return m_cfg.num_kv_heads * m_cfg.head_dim; }
    size_t total_elements_per_cache() const {
        return static_cast<size_t>(m_cfg.num_layers) * m_cfg.max_seq_len * kv_dim();
    }
    size_t total_bytes() const {
        return total_elements_per_cache() * sizeof(float) * 2; // K + V
    }

    // Pointer accessors
    float* get_k_layer_pos(uint32_t layer, uint32_t pos);
    float* get_v_layer_pos(uint32_t layer, uint32_t pos);

    const float* get_k_layer_pos(uint32_t layer, uint32_t pos) const;
    const float* get_v_layer_pos(uint32_t layer, uint32_t pos) const;

    // Put current token K and V at position
    void put(uint32_t layer, uint32_t pos, const float *k_src, const float *v_src);

    const KvCacheConfig& config() const { return m_cfg; }

private:
    KvCacheConfig      m_cfg;
    std::vector<float> m_k_cache;
    std::vector<float> m_v_cache;
};

} // namespace eswin

#endif // ESWIN_KV_CACHE_HPP
