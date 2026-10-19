"""
runtime.py - High-level Pythonic wrapper for Open ESWIN NPU Runtime
"""

import ctypes
from enum import IntEnum
from typing import Optional, Callable, Generator, NamedTuple

from .binding import (
    load_lib,
    ES_SUCCESS,
    ES_DEVICE_DIE0,
    ES_DEVICE_DIE1,
    ES_DEVICE_DUAL_DIE_AUTO,
    ES_MEM_NOCACHE,
    ES_MEM_CACHED,
    es_context_t,
    es_dma_buf_t,
    es_model_t,
    es_session_t,
    es_tokenizer_t,
    es_sampler_t,
    es_model_config_t,
    es_model_info_t,
    es_session_config_t,
    es_sampler_params_t,
    es_perf_stats_t,
    es_token_callback_t,
)

_LIB = None

def get_lib():
    global _LIB
    if _LIB is None:
        _LIB = load_lib()
    return _LIB


def version() -> str:
    return get_lib().es_version().decode("utf-8")


class DeviceMode(IntEnum):
    DIE0 = ES_DEVICE_DIE0
    DIE1 = ES_DEVICE_DIE1
    DUAL_DIE_AUTO = ES_DEVICE_DUAL_DIE_AUTO


class CacheMode(IntEnum):
    NOCACHE = ES_MEM_NOCACHE
    CACHED = ES_MEM_CACHED


class PerfStats(NamedTuple):
    prompt_tokens: int
    eval_tokens: int
    prompt_eval_ms: float
    eval_ms: float
    tokens_per_sec: float


class ModelInfo(NamedTuple):
    name: str
    architecture: str
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    max_seq_len: int


class SamplerParams:
    def __init__(
        self,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        min_p: float = 0.05,
        repeat_penalty: float = 1.15,
        repeat_window: int = 64,
        max_consecutive_duplicates: int = 2,
        seed: int = 42,
    ):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.repeat_penalty = repeat_penalty
        self.repeat_window = repeat_window
        self.max_consecutive_duplicates = max_consecutive_duplicates
        self.seed = seed

    def to_c(self) -> es_sampler_params_t:
        c_params = es_sampler_params_t()
        c_params.temperature = self.temperature
        c_params.top_k = self.top_k
        c_params.top_p = self.top_p
        c_params.min_p = self.min_p
        c_params.repeat_penalty = self.repeat_penalty
        c_params.repeat_window = self.repeat_window
        c_params.max_consecutive_duplicates = self.max_consecutive_duplicates
        c_params.seed = self.seed
        return c_params


class DmaBuffer:
    def __init__(self, ctx_handle, size: int, cache_mode: CacheMode = CacheMode.NOCACHE):
        self._lib = get_lib()
        self._handle = ctypes.POINTER(es_dma_buf_t)()
        status = self._lib.es_dma_alloc(ctx_handle, size, cache_mode.value, ctypes.byref(self._handle))
        if status != ES_SUCCESS:
            err = self._lib.es_status_to_string(status).decode("utf-8")
            raise RuntimeError(f"es_dma_alloc failed: {err}")
        self._size = size

    @property
    def size(self) -> int:
        return self._lib.es_dma_get_size(self._handle)

    @property
    def mem_fd(self) -> int:
        return self._lib.es_dma_get_fd(self._handle)

    @property
    def host_ptr(self) -> int:
        return self._lib.es_dma_get_host_ptr(self._handle)

    def flush_cache(self):
        status = self._lib.es_dma_flush_cache(self._handle)
        if status != ES_SUCCESS:
            err = self._lib.es_status_to_string(status).decode("utf-8")
            raise RuntimeError(f"es_dma_flush_cache failed: {err}")

    def close(self):
        if self._handle:
            self._lib.es_dma_free(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


class RuntimeContext:
    def __init__(self):
        self._lib = get_lib()
        self._handle = ctypes.POINTER(es_context_t)()
        status = self._lib.es_init(ctypes.byref(self._handle))
        if status != ES_SUCCESS:
            err = self._lib.es_status_to_string(status).decode("utf-8")
            raise RuntimeError(f"es_init failed: {err}")

    @property
    def device_count(self) -> int:
        count = ctypes.c_uint16()
        status = self._lib.es_get_device_count(self._handle, ctypes.byref(count))
        if status != ES_SUCCESS:
            err = self._lib.es_status_to_string(status).decode("utf-8")
            raise RuntimeError(f"es_get_device_count failed: {err}")
        return count.value

    def allocate_dma(self, size: int, cache_mode: CacheMode = CacheMode.NOCACHE) -> DmaBuffer:
        return DmaBuffer(self._handle, size, cache_mode)

    def load_model(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device_mode: DeviceMode = DeviceMode.DUAL_DIE_AUTO,
    ) -> "Model":
        return Model(self, model_path, tokenizer_path, cache_dir, device_mode)

    def close(self):
        if self._handle:
            self._lib.es_free(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


class Model:
    def __init__(
        self,
        ctx: RuntimeContext,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device_mode: DeviceMode = DeviceMode.DUAL_DIE_AUTO,
    ):
        self._lib = get_lib()
        self._ctx = ctx
        self._handle = ctypes.POINTER(es_model_t)()

        cfg = es_model_config_t()
        cfg.model_path = model_path.encode("utf-8")
        cfg.tokenizer_path = tokenizer_path.encode("utf-8") if tokenizer_path else None
        cfg.cache_dir = cache_dir.encode("utf-8") if cache_dir else None
        cfg.device_mode = device_mode.value

        status = self._lib.es_model_load(ctx._handle, ctypes.byref(cfg), ctypes.byref(self._handle))
        if status != ES_SUCCESS:
            err = self._lib.es_status_to_string(status).decode("utf-8")
            raise RuntimeError(f"es_model_load failed: {err}")

        # Fetch metadata
        c_info = es_model_info_t()
        status = self._lib.es_model_get_info(self._handle, ctypes.byref(c_info))
        if status == ES_SUCCESS:
            self._info = ModelInfo(
                name=c_info.name.decode("utf-8", errors="ignore"),
                architecture=c_info.architecture.decode("utf-8", errors="ignore"),
                vocab_size=c_info.vocab_size,
                hidden_size=c_info.hidden_size,
                num_layers=c_info.num_layers,
                num_heads=c_info.num_heads,
                num_kv_heads=c_info.num_kv_heads,
                max_seq_len=c_info.max_seq_len,
            )
        else:
            self._info = None

    @property
    def info(self) -> Optional[ModelInfo]:
        return self._info

    def create_session(
        self,
        max_seq_len: int = 2048,
        max_new_tokens: int = 512,
        enable_dual_die: bool = True,
        sampler_params: Optional[SamplerParams] = None,
    ) -> "Session":
        return Session(self, max_seq_len, max_new_tokens, enable_dual_die, sampler_params)

    def close(self):
        if self._handle:
            self._lib.es_model_unload(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


class Session:
    def __init__(
        self,
        model: Model,
        max_seq_len: int = 2048,
        max_new_tokens: int = 512,
        enable_dual_die: bool = True,
        sampler_params: Optional[SamplerParams] = None,
    ):
        self._lib = get_lib()
        self._model = model
        self._handle = ctypes.POINTER(es_session_t)()

        cfg = es_session_config_t()
        cfg.max_seq_len = max_seq_len
        cfg.max_new_tokens = max_new_tokens
        cfg.enable_dual_die = enable_dual_die
        if sampler_params is None:
            sampler_params = SamplerParams()
        cfg.sampler_params = sampler_params.to_c()

        status = self._lib.es_session_create(model._handle, ctypes.byref(cfg), ctypes.byref(self._handle))
        if status != ES_SUCCESS:
            err = self._lib.es_status_to_string(status).decode("utf-8")
            raise RuntimeError(f"es_session_create failed: {err}")

        self._last_stats = None

    @property
    def last_stats(self) -> Optional[PerfStats]:
        return self._last_stats

    def reset(self):
        status = self._lib.es_session_reset(self._handle)
        if status != ES_SUCCESS:
            err = self._lib.es_status_to_string(status).decode("utf-8")
            raise RuntimeError(f"es_session_reset failed: {err}")

    def generate(
        self,
        prompt: str,
        callback: Optional[Callable[[int, str], bool]] = None,
    ) -> str:
        collected = []

        def c_callback(token_id: int, piece_ptr: bytes, user_data: ctypes.c_void_p) -> bool:
            piece = piece_ptr.decode("utf-8", errors="ignore") if piece_ptr else ""
            collected.append(piece)
            if callback:
                return callback(token_id, piece)
            return True

        cb_fn = es_token_callback_t(c_callback)
        c_stats = es_perf_stats_t()

        status = self._lib.es_session_generate(
            self._handle,
            prompt.encode("utf-8"),
            cb_fn,
            None,
            ctypes.byref(c_stats)
        )

        if status != ES_SUCCESS:
            err = self._lib.es_status_to_string(status).decode("utf-8")
            raise RuntimeError(f"es_session_generate failed: {err}")

        self._last_stats = PerfStats(
            prompt_tokens=c_stats.prompt_tokens,
            eval_tokens=c_stats.eval_tokens,
            prompt_eval_ms=c_stats.prompt_eval_ms,
            eval_ms=c_stats.eval_ms,
            tokens_per_sec=c_stats.tokens_per_sec,
        )

        return "".join(collected)

    def close(self):
        if self._handle:
            self._lib.es_session_free(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()
