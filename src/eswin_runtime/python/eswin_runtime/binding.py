"""
binding.py - ctypes foreign function interface to libes_runtime.so
"""

import os
import sys
import ctypes
from ctypes import (
    c_int, c_uint16, c_uint32, c_uint64, c_size_t, c_float, c_double,
    c_char, c_char_p, c_void_p, c_bool, Structure, POINTER, CFUNCTYPE
)

# Status Enum
ES_SUCCESS               = 0
ES_ERR_INVALID_ARG       = -1
ES_ERR_NO_MEMORY         = -2
ES_ERR_DEVICE_BUSY       = -3
ES_ERR_HARDWARE_FAULT    = -4
ES_ERR_NOT_INITIALIZED   = -5
ES_ERR_NOT_SUPPORTED     = -6
ES_ERR_IO                = -7
ES_ERR_CHECKSUM_MISMATCH = -8
ES_ERR_TIMEOUT           = -9

# Device Mode
ES_DEVICE_DIE0          = 0
ES_DEVICE_DIE1          = 1
ES_DEVICE_DUAL_DIE_AUTO = 2

# Cache Mode
ES_MEM_NOCACHE          = 0
ES_MEM_CACHED           = 1

class es_context_t(Structure):
    pass

class es_dma_buf_t(Structure):
    pass

class es_model_t(Structure):
    pass

class es_session_t(Structure):
    pass

class es_tokenizer_t(Structure):
    pass

class es_sampler_t(Structure):
    pass

class es_sampler_params_t(Structure):
    _fields_ = [
        ("temperature", c_float),
        ("top_k", c_int),
        ("top_p", c_float),
        ("min_p", c_float),
        ("repeat_penalty", c_float),
        ("repeat_window", c_int),
        ("max_consecutive_duplicates", c_int),
        ("seed", c_uint32),
    ]

class es_model_config_t(Structure):
    _fields_ = [
        ("model_path", c_char_p),
        ("tokenizer_path", c_char_p),
        ("cache_dir", c_char_p),
        ("device_mode", c_int),
    ]

class es_model_info_t(Structure):
    _fields_ = [
        ("name", c_char * 128),
        ("architecture", c_char * 64),
        ("vocab_size", c_uint32),
        ("hidden_size", c_uint32),
        ("num_layers", c_uint32),
        ("num_heads", c_uint32),
        ("num_kv_heads", c_uint32),
        ("max_seq_len", c_uint32),
    ]

class es_session_config_t(Structure):
    _fields_ = [
        ("max_seq_len", c_size_t),
        ("max_new_tokens", c_int),
        ("enable_dual_die", c_bool),
        ("sampler_params", es_sampler_params_t),
    ]

class es_perf_stats_t(Structure):
    _fields_ = [
        ("prompt_tokens", c_int),
        ("eval_tokens", c_int),
        ("prompt_eval_ms", c_double),
        ("eval_ms", c_double),
        ("tokens_per_sec", c_double),
    ]

# Callback: bool (*)(int32_t token, const char *piece, void *user_data)
es_token_callback_t = CFUNCTYPE(c_bool, c_int, c_char_p, c_void_p)


def find_library():
    # 1. Environment variable
    env_lib = os.environ.get("ES_RUNTIME_LIB")
    if env_lib and os.path.exists(env_lib):
        return env_lib

    # 2. Relative paths from package
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(current_dir, "libes_runtime.so"),
        os.path.join(current_dir, "..", "libes_runtime.so"),
        os.path.join(current_dir, "..", "..", "build", "libes_runtime.so"),
        os.path.join(current_dir, "..", "..", "..", "build", "libes_runtime.so"),
        "/usr/local/lib/libes_runtime.so",
        "/usr/lib/libes_runtime.so",
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)

    return "libes_runtime.so"


def load_lib():
    lib_path = find_library()
    try:
        lib = ctypes.CDLL(lib_path)
    except OSError as e:
        raise RuntimeError(
            f"Failed to load libes_runtime.so from '{lib_path}'. "
            "Ensure the runtime is built (make) or set ES_RUNTIME_LIB."
        ) from e

    # Version & Status
    lib.es_version.restype = c_char_p
    lib.es_version.argtypes = []

    lib.es_status_to_string.restype = c_char_p
    lib.es_status_to_string.argtypes = [c_int]

    # Context
    lib.es_init.restype = c_int
    lib.es_init.argtypes = [POINTER(POINTER(es_context_t))]

    lib.es_free.restype = None
    lib.es_free.argtypes = [POINTER(es_context_t)]

    lib.es_get_device_count.restype = c_int
    lib.es_get_device_count.argtypes = [POINTER(es_context_t), POINTER(c_uint16)]

    # DMA
    lib.es_dma_alloc.restype = c_int
    lib.es_dma_alloc.argtypes = [POINTER(es_context_t), c_size_t, c_int, POINTER(POINTER(es_dma_buf_t))]

    lib.es_dma_free.restype = None
    lib.es_dma_free.argtypes = [POINTER(es_dma_buf_t)]

    lib.es_dma_get_host_ptr.restype = c_void_p
    lib.es_dma_get_host_ptr.argtypes = [POINTER(es_dma_buf_t)]

    lib.es_dma_get_fd.restype = c_uint64
    lib.es_dma_get_fd.argtypes = [POINTER(es_dma_buf_t)]

    lib.es_dma_get_size.restype = c_size_t
    lib.es_dma_get_size.argtypes = [POINTER(es_dma_buf_t)]

    lib.es_dma_flush_cache.restype = c_int
    lib.es_dma_flush_cache.argtypes = [POINTER(es_dma_buf_t)]

    # Tokenizer
    lib.es_tokenizer_load.restype = c_int
    lib.es_tokenizer_load.argtypes = [c_char_p, POINTER(POINTER(es_tokenizer_t))]

    lib.es_tokenizer_free.restype = None
    lib.es_tokenizer_free.argtypes = [POINTER(es_tokenizer_t)]

    lib.es_tokenizer_encode.restype = c_int
    lib.es_tokenizer_encode.argtypes = [POINTER(es_tokenizer_t), c_char_p, POINTER(c_int), c_size_t]

    lib.es_tokenizer_decode.restype = c_int
    lib.es_tokenizer_decode.argtypes = [POINTER(es_tokenizer_t), POINTER(c_int), c_size_t, c_char_p, c_size_t]

    # Sampler
    lib.es_sampler_create.restype = c_int
    lib.es_sampler_create.argtypes = [POINTER(es_sampler_params_t), POINTER(POINTER(es_sampler_t))]

    lib.es_sampler_free.restype = None
    lib.es_sampler_free.argtypes = [POINTER(es_sampler_t)]

    lib.es_sampler_sample.restype = c_int
    lib.es_sampler_sample.argtypes = [POINTER(es_sampler_t), POINTER(c_float), c_size_t]

    lib.es_sampler_accept.restype = None
    lib.es_sampler_accept.argtypes = [POINTER(es_sampler_t), c_int]

    lib.es_sampler_reset.restype = None
    lib.es_sampler_reset.argtypes = [POINTER(es_sampler_t)]

    # Model
    lib.es_model_load.restype = c_int
    lib.es_model_load.argtypes = [POINTER(es_context_t), POINTER(es_model_config_t), POINTER(POINTER(es_model_t))]

    lib.es_model_unload.restype = None
    lib.es_model_unload.argtypes = [POINTER(es_model_t)]

    lib.es_model_get_info.restype = c_int
    lib.es_model_get_info.argtypes = [POINTER(es_model_t), POINTER(es_model_info_t)]

    lib.es_model_get_tokenizer.restype = POINTER(es_tokenizer_t)
    lib.es_model_get_tokenizer.argtypes = [POINTER(es_model_t)]

    # Session
    lib.es_session_create.restype = c_int
    lib.es_session_create.argtypes = [POINTER(es_model_t), POINTER(es_session_config_t), POINTER(POINTER(es_session_t))]

    lib.es_session_free.restype = None
    lib.es_session_free.argtypes = [POINTER(es_session_t)]

    lib.es_session_reset.restype = c_int
    lib.es_session_reset.argtypes = [POINTER(es_session_t)]

    lib.es_session_generate.restype = c_int
    lib.es_session_generate.argtypes = [
        POINTER(es_session_t),
        c_char_p,
        es_token_callback_t,
        c_void_p,
        POINTER(es_perf_stats_t)
    ]

    return lib
