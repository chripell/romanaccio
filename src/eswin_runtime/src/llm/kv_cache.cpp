#include "kv_cache.hpp"
#include <algorithm>
#include <cstring>

namespace eswin {

KvCache::KvCache(const KvCacheConfig &cfg)
    : m_cfg(cfg)
{
    size_t count = total_elements_per_cache();
    m_k_cache.assign(count, 0.0f);
    m_v_cache.assign(count, 0.0f);
}

void KvCache::reset() {
    std::fill(m_k_cache.begin(), m_k_cache.end(), 0.0f);
    std::fill(m_v_cache.begin(), m_v_cache.end(), 0.0f);
}

float* KvCache::get_k_layer_pos(uint32_t layer, uint32_t pos) {
    if (layer >= m_cfg.num_layers || pos >= m_cfg.max_seq_len) return nullptr;
    size_t offset = (static_cast<size_t>(layer) * m_cfg.max_seq_len + pos) * kv_dim();
    return m_k_cache.data() + offset;
}

float* KvCache::get_v_layer_pos(uint32_t layer, uint32_t pos) {
    if (layer >= m_cfg.num_layers || pos >= m_cfg.max_seq_len) return nullptr;
    size_t offset = (static_cast<size_t>(layer) * m_cfg.max_seq_len + pos) * kv_dim();
    return m_v_cache.data() + offset;
}

const float* KvCache::get_k_layer_pos(uint32_t layer, uint32_t pos) const {
    if (layer >= m_cfg.num_layers || pos >= m_cfg.max_seq_len) return nullptr;
    size_t offset = (static_cast<size_t>(layer) * m_cfg.max_seq_len + pos) * kv_dim();
    return m_k_cache.data() + offset;
}

const float* KvCache::get_v_layer_pos(uint32_t layer, uint32_t pos) const {
    if (layer >= m_cfg.num_layers || pos >= m_cfg.max_seq_len) return nullptr;
    size_t offset = (static_cast<size_t>(layer) * m_cfg.max_seq_len + pos) * kv_dim();
    return m_v_cache.data() + offset;
}

void KvCache::put(uint32_t layer, uint32_t pos, const float *k_src, const float *v_src) {
    float *k_dst = get_k_layer_pos(layer, pos);
    float *v_dst = get_v_layer_pos(layer, pos);
    size_t bytes = kv_dim() * sizeof(float);
    if (k_dst && k_src) std::memcpy(k_dst, k_src, bytes);
    if (v_dst && v_src) std::memcpy(v_dst, v_src, bytes);
}

} // namespace eswin
