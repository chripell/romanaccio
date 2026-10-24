/**
 * @file hardware_kv_cache.cpp
 * @brief Physically contiguous MMZ KV-cache with per-package layout (Milestone 9).
 *
 * See hardware_kv_cache.hpp for the two supported layouts and the evidence behind them.
 *
 * Buffers are CacheMode::Cached even though the NPU is the primary writer: the CPU reads
 * them back for golden-data verification and in host fallback, and NoCache MMZ reads
 * collapse to 10-60 MB/s versus 1.74-2.89 GB/s cached. Every CPU write is followed by
 * flush_cache() so the NPU sees it.
 */

#include "hardware_kv_cache.hpp"
#include "../model/embedding_table.hpp"   // es_f32_to_f16 / es_f16_to_f32

#include <cstring>
#include <sstream>

namespace eswin {

namespace {

const char *zone_for_die(uint16_t die) {
    return (die == 1) ? "mmz_nid_1_part_0" : "mmz_nid_0_part_0";
}

} // namespace

uint32_t HardwareKvCache::default_token_n2(uint32_t max_seq_len) noexcept {
    static const uint32_t kCandidates[] = {256, 128, 64, 32, 16, 8, 4, 2, 1};
    for (uint32_t c : kCandidates) {
        if (max_seq_len % c == 0) return c;
    }
    return 1;
}

HardwareKvCache::~HardwareKvCache() {
    release();
}

size_t HardwareKvCache::bytes_per_layer() const noexcept {
    return kv_dim() * static_cast<size_t>(m_cfg.max_seq_len) * m_cfg.elem_bytes;
}

uint16_t HardwareKvCache::die_for_layer(uint32_t layer) const noexcept {
    return (layer < m_layer_die.size()) ? m_layer_die[layer] : 0;
}

bool HardwareKvCache::allocate(const HardwareKvConfig &cfg, WorkerPool *pool, std::string &err) {
    release();
    err.clear();

    if (cfg.num_layers == 0 || cfg.num_kv_heads == 0 || cfg.head_dim == 0 ||
        cfg.max_seq_len == 0) {
        err = "HardwareKvCache: num_layers, num_kv_heads, head_dim and max_seq_len "
              "must all be non-zero";
        return false;
    }
    if (cfg.elem_bytes != 2 && cfg.elem_bytes != 4) {
        err = "HardwareKvCache: elem_bytes must be 2 (fp16) or 4 (fp32), got " +
              std::to_string(cfg.elem_bytes);
        return false;
    }

    m_cfg = cfg;
    if (cfg.layout == KvLayout::TokenMajorTiled) {
        // The tile height is a compile-time property of the loadable, published by the
        // package as token_n2 (16 for Qwen2-0.5B 1k INT8). A divisor heuristic picks 256
        // there, and a wrong tile height mis-addresses every V write without crashing -
        // it just produces Issue #47-style corruption. Refuse to guess.
        if (cfg.token_n2 == 0) {
            err = "HardwareKvCache: TokenMajorTiled requires an explicit token_n2 taken "
                  "from the package (ModelConfig::kv_token_n2); refusing to guess a "
                  "V-cache tile height";
            return false;
        }
        m_token_n2 = cfg.token_n2;
    } else {
        m_token_n2 = cfg.token_n2 ? cfg.token_n2 : default_token_n2(cfg.max_seq_len);
    }
    if (m_token_n2 == 0 || cfg.max_seq_len % m_token_n2 != 0) {
        err = "HardwareKvCache: token_n2 (" + std::to_string(m_token_n2) +
              ") must divide max_seq_len (" + std::to_string(cfg.max_seq_len) + ")";
        return false;
    }

    const size_t per_layer = bytes_per_layer();
    if (per_layer == 0) {
        err = "HardwareKvCache: computed a zero-byte layer cache";
        return false;
    }

    m_pool = pool;
    m_layer_die.assign(cfg.num_layers, 0);
    m_k.reserve(cfg.num_layers);
    m_v.reserve(cfg.num_layers);

    for (uint32_t layer = 0; layer < cfg.num_layers; ++layer) {
        // Pipeline-parallel split: first half of the layers on Die 0, rest on Die 1.
        const uint16_t die = (cfg.dual_die && layer >= cfg.num_layers / 2) ? 1 : 0;
        m_layer_die[layer] = die;

        auto k = std::make_unique<DmaBuffer>(per_layer, CacheMode::Cached, zone_for_die(die));
        auto v = std::make_unique<DmaBuffer>(per_layer, CacheMode::Cached, zone_for_die(die));
        if (!k->is_valid() || !v->is_valid()) {
            std::ostringstream os;
            os << "HardwareKvCache: allocation failed at layer " << layer << " ("
               << per_layer << " bytes per cache)";
            err = os.str();
            release();
            return false;
        }

        if (pool) {
            // Must be prepared inside the worker that submits, or ES_NPU_Prepare aborts
            // with 0xA00F604A (no thread-local device context).
            if (!pool->prepare_buffer(die, *k) || !pool->prepare_buffer(die, *v)) {
                std::ostringstream os;
                os << "HardwareKvCache: SMMU prepare failed for layer " << layer
                   << " on die " << die;
                err = os.str();
                release();
                return false;
            }
        }

        m_k.push_back(std::move(k));
        m_v.push_back(std::move(v));
        m_total_bytes += 2 * per_layer;
    }

    m_length = 0;
    return true;
}

void HardwareKvCache::release() {
    for (size_t i = 0; i < m_k.size(); ++i) {
        const uint16_t die = (i < m_layer_die.size()) ? m_layer_die[i] : 0;
        if (m_pool) {
            if (m_k[i]) m_pool->unprepare_buffer(die, *m_k[i]);
            if (i < m_v.size() && m_v[i]) m_pool->unprepare_buffer(die, *m_v[i]);
        }
    }
    m_k.clear();
    m_v.clear();
    m_layer_die.clear();
    m_length = 0;
    m_total_bytes = 0;
    m_pool = nullptr;
}

uint32_t HardwareKvCache::slot_for(uint32_t token) const noexcept {
    if (m_cfg.max_seq_len == 0) return 0;
    return m_cfg.wrap ? (token % m_cfg.max_seq_len) : token;
}

size_t HardwareKvCache::k_offset(uint32_t head, uint32_t dim, uint32_t token) const noexcept {
    const size_t t = slot_for(token);
    const size_t e = m_cfg.elem_bytes;
    if (m_cfg.layout == KvLayout::HeadMajor) {
        // [head][dim][token]: token is the contiguous axis, as in (2, 64, 1, 1024).
        return ((static_cast<size_t>(head) * m_cfg.head_dim + dim) *
                    static_cast<size_t>(m_cfg.max_seq_len) + t) * e;
    }
    // [token][kv_dim]: the FINDINGS section 5 K formula, elem * kv_dim * token.
    return (t * kv_dim() + static_cast<size_t>(head) * m_cfg.head_dim + dim) * e;
}

size_t HardwareKvCache::v_offset(uint32_t head, uint32_t token, uint32_t dim) const noexcept {
    const size_t t = slot_for(token);
    const size_t e = m_cfg.elem_bytes;
    if (m_cfg.layout == KvLayout::HeadMajor) {
        // [head][token][dim]: dim is the contiguous axis, as in (2, 1024, 1, 64).
        return ((static_cast<size_t>(head) * m_cfg.max_seq_len + t) * m_cfg.head_dim + dim) * e;
    }
    // Tiled: FINDINGS section 5 V formula, extended with the per-head and per-dim terms.
    const size_t n2 = m_token_n2;
    const size_t tiles = m_cfg.max_seq_len / n2;
    const size_t head_stride = static_cast<size_t>(m_cfg.head_dim) * tiles * n2;
    return (static_cast<size_t>(head) * head_stride +
            n2 * static_cast<size_t>(m_cfg.head_dim) * (t / n2) +
            static_cast<size_t>(dim) * n2 +
            (t % n2)) * e;
}

size_t HardwareKvCache::k_token_offset(uint32_t token) const noexcept {
    if (m_cfg.layout == KvLayout::TokenMajorTiled) {
        // Vendor K formula verbatim: precision * kv_dim * token_idx.
        return m_cfg.elem_bytes * kv_dim() * slot_for(token);
    }
    return k_offset(0, 0, token);
}

size_t HardwareKvCache::v_token_offset(uint32_t token) const noexcept {
    const size_t t = slot_for(token);
    if (m_cfg.layout == KvLayout::TokenMajorTiled) {
        // Vendor V formula verbatim:
        // precision * (token_n2 * head_dim * (t / token_n2) + (t % token_n2)).
        return m_cfg.elem_bytes * (static_cast<size_t>(m_token_n2) * m_cfg.head_dim *
                                       (t / m_token_n2) + (t % m_token_n2));
    }
    return v_offset(0, static_cast<uint32_t>(t), 0);
}

bool HardwareKvCache::advance(uint32_t n) {
    if (!is_allocated()) return false;
    if (!m_cfg.wrap && m_length + n > m_cfg.max_seq_len) {
        // Strict mode: refuse rather than write past the compiled window.
        return false;
    }
    const uint64_t next = static_cast<uint64_t>(m_length) + n;
    m_length = (next > m_cfg.max_seq_len) ? m_cfg.max_seq_len : static_cast<uint32_t>(next);
    return true;
}

void HardwareKvCache::reset(bool zero_fill) {
    m_length = 0;
    if (!zero_fill) return;
    for (size_t i = 0; i < m_k.size(); ++i) {
        if (m_k[i] && m_k[i]->is_valid()) {
            std::memset(m_k[i]->data(), 0, m_k[i]->size());
            m_k[i]->flush_cache();
        }
        if (i < m_v.size() && m_v[i] && m_v[i]->is_valid()) {
            std::memset(m_v[i]->data(), 0, m_v[i]->size());
            m_v[i]->flush_cache();
        }
    }
}

DmaBuffer *HardwareKvCache::k_buffer(uint32_t layer) {
    return (layer < m_k.size()) ? m_k[layer].get() : nullptr;
}

DmaBuffer *HardwareKvCache::v_buffer(uint32_t layer) {
    return (layer < m_v.size()) ? m_v[layer].get() : nullptr;
}

const DmaBuffer *HardwareKvCache::k_buffer(uint32_t layer) const {
    return (layer < m_k.size()) ? m_k[layer].get() : nullptr;
}

const DmaBuffer *HardwareKvCache::v_buffer(uint32_t layer) const {
    return (layer < m_v.size()) ? m_v[layer].get() : nullptr;
}

uint64_t HardwareKvCache::k_mem_fd(uint32_t layer) const {
    const DmaBuffer *b = k_buffer(layer);
    return b ? b->mem_fd() : 0;
}

uint64_t HardwareKvCache::v_mem_fd(uint32_t layer) const {
    const DmaBuffer *b = v_buffer(layer);
    return b ? b->mem_fd() : 0;
}

const void *HardwareKvCache::k_elem(uint32_t layer, uint32_t head, uint32_t dim,
                                    uint32_t token) const {
    const DmaBuffer *b = k_buffer(layer);
    if (!b || head >= m_cfg.num_kv_heads || dim >= m_cfg.head_dim) return nullptr;
    const size_t off = k_offset(head, dim, token);
    if (off + m_cfg.elem_bytes > b->size()) return nullptr;
    return static_cast<const char *>(b->data()) + off;
}

const void *HardwareKvCache::v_elem(uint32_t layer, uint32_t head, uint32_t token,
                                    uint32_t dim) const {
    const DmaBuffer *b = v_buffer(layer);
    if (!b || head >= m_cfg.num_kv_heads || dim >= m_cfg.head_dim) return nullptr;
    const size_t off = v_offset(head, token, dim);
    if (off + m_cfg.elem_bytes > b->size()) return nullptr;
    return static_cast<const char *>(b->data()) + off;
}

void *HardwareKvCache::k_elem_mut(uint32_t layer, uint32_t head, uint32_t dim,
                                  uint32_t token) {
    DmaBuffer *b = k_buffer(layer);
    if (!b || head >= m_cfg.num_kv_heads || dim >= m_cfg.head_dim) return nullptr;
    const size_t off = k_offset(head, dim, token);
    if (off + m_cfg.elem_bytes > b->size()) return nullptr;
    return static_cast<char *>(b->data()) + off;
}

void *HardwareKvCache::v_elem_mut(uint32_t layer, uint32_t head, uint32_t token,
                                  uint32_t dim) {
    DmaBuffer *b = v_buffer(layer);
    if (!b || head >= m_cfg.num_kv_heads || dim >= m_cfg.head_dim) return nullptr;
    const size_t off = v_offset(head, token, dim);
    if (off + m_cfg.elem_bytes > b->size()) return nullptr;
    return static_cast<char *>(b->data()) + off;
}

namespace {

void store_elem(void *dst, float value, size_t elem_bytes) {
    if (elem_bytes == 4) {
        std::memcpy(dst, &value, sizeof(float));
    } else {
        const uint16_t h = es_f32_to_f16(value);
        std::memcpy(dst, &h, sizeof(h));
    }
}

float load_elem(const void *src, size_t elem_bytes) {
    if (elem_bytes == 4) {
        float f;
        std::memcpy(&f, src, sizeof(f));
        return f;
    }
    uint16_t h;
    std::memcpy(&h, src, sizeof(h));
    return es_f16_to_f32(h);
}

} // namespace

bool HardwareKvCache::write_k(uint32_t layer, uint32_t token, const float *values) {
    if (!values || layer >= m_k.size()) return false;
    for (uint32_t h = 0; h < m_cfg.num_kv_heads; ++h) {
        for (uint32_t d = 0; d < m_cfg.head_dim; ++d) {
            void *p = k_elem_mut(layer, h, d, token);
            if (!p) return false;
            store_elem(p, values[static_cast<size_t>(h) * m_cfg.head_dim + d], m_cfg.elem_bytes);
        }
    }
    if (m_k[layer]) m_k[layer]->flush_cache();
    return true;
}

bool HardwareKvCache::write_v(uint32_t layer, uint32_t token, const float *values) {
    if (!values || layer >= m_v.size()) return false;
    for (uint32_t h = 0; h < m_cfg.num_kv_heads; ++h) {
        for (uint32_t d = 0; d < m_cfg.head_dim; ++d) {
            void *p = v_elem_mut(layer, h, token, d);
            if (!p) return false;
            store_elem(p, values[static_cast<size_t>(h) * m_cfg.head_dim + d], m_cfg.elem_bytes);
        }
    }
    if (m_v[layer]) m_v[layer]->flush_cache();
    return true;
}

bool HardwareKvCache::read_k(uint32_t layer, uint32_t token, float *out) const {
    if (!out || layer >= m_k.size()) return false;
    for (uint32_t h = 0; h < m_cfg.num_kv_heads; ++h) {
        for (uint32_t d = 0; d < m_cfg.head_dim; ++d) {
            const void *p = k_elem(layer, h, d, token);
            if (!p) return false;
            out[static_cast<size_t>(h) * m_cfg.head_dim + d] = load_elem(p, m_cfg.elem_bytes);
        }
    }
    return true;
}

bool HardwareKvCache::read_v(uint32_t layer, uint32_t token, float *out) const {
    if (!out || layer >= m_v.size()) return false;
    for (uint32_t h = 0; h < m_cfg.num_kv_heads; ++h) {
        for (uint32_t d = 0; d < m_cfg.head_dim; ++d) {
            const void *p = v_elem(layer, h, token, d);
            if (!p) return false;
            out[static_cast<size_t>(h) * m_cfg.head_dim + d] = load_elem(p, m_cfg.elem_bytes);
        }
    }
    return true;
}

} // namespace eswin
