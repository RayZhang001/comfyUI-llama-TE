# -*- coding: utf-8 -*-
"""VRAM coordination for the patched Qwen llama.cpp backend.

Before a real Qwen model load, optionally unload ComfyUI-managed models and
clear CUDA/PyTorch caches so llama.cpp starts with as much of the GPU as
possible. Cached Qwen reuse skips the cleanup.

This module also suppresses the native llama.cpp verbosity=3 dump added by the
performance diagnostics shim while keeping the concise custom verbose metrics.
"""

from __future__ import annotations

import gc
import inspect
import time

try:
    import torch
except Exception:
    torch = None

from . import nodes as _nodes
from . import qwen_perf_patch as _perf


if not getattr(_nodes, "_qwen_vram_patch_installed", False):
    _PARENT_LLAMA = _nodes.Llama
    _PARENT_QWEN_LOAD = _nodes._QwenStorage.load.__func__
    _PARENT_MODEL_LOADER_INPUT_TYPES = _nodes.QwenTE模型加载器.INPUT_TYPES
    _PARENT_MODEL_LOADER_LOAD = _nodes.QwenTE模型加载器.__dict__["load"]

    _PENDING_FREE_COMFY_VRAM = None

    def _gpu_mem_mib():
        if torch is None or not torch.cuda.is_available():
            return None
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            mib = 1024.0 * 1024.0
            return free_bytes / mib, total_bytes / mib
        except Exception:
            return None

    def _format_gpu_mem(mem):
        if mem is None:
            return "unavailable"
        free_mib, total_mib = mem
        return f"free={free_mib:.0f} MiB, total={total_mib:.0f} MiB"

    def _force_free_comfy_vram(verbose_logging: bool) -> None:
        before = _gpu_mem_mib()
        started = time.perf_counter()

        if verbose_logging:
            print(
                "[comfyUI-llama-TE verbose] Pre-Qwen VRAM cleanup start: "
                + _format_gpu_mem(before),
                flush=True,
            )

        # Finish any outstanding CUDA work before model teardown, then release
        # ComfyUI-owned models and allocator caches. The Qwen unload hook is
        # safe here and also closes a stale Qwen instance when configuration
        # changed between runs.
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

        try:
            _nodes.mm.unload_all_models()
        except Exception as exc:
            raise RuntimeError(
                "Failed to unload ComfyUI models before loading Qwen."
            ) from exc

        gc.collect()

        try:
            _nodes.mm.soft_empty_cache()
        except Exception:
            pass

        if torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

        after = _gpu_mem_mib()
        if verbose_logging:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            reclaimed_text = ""
            if before is not None and after is not None:
                reclaimed_text = f", reclaimed={after[0] - before[0]:.0f} MiB"
            print(
                "[comfyUI-llama-TE verbose] Pre-Qwen VRAM cleanup complete: "
                f"{_format_gpu_mem(after)}{reclaimed_text}, wall_ms={elapsed_ms:.2f}",
                flush=True,
            )

    @classmethod
    def _qwen_vram_input_types(cls):
        schema = _PARENT_MODEL_LOADER_INPUT_TYPES()
        schema["required"]["Free ComfyUI VRAM Before Qwen"] = (
            "BOOLEAN",
            {
                "default": True,
                "tooltip": (
                    "Before loading a new Qwen instance, unload ComfyUI-managed models "
                    "and clear CUDA caches so llama.cpp can use the available VRAM. "
                    "Cached Qwen reuse is not disturbed."
                ),
            },
        )
        return schema

    def _qwen_vram_model_loader_load(self, *args, **kwargs):
        global _PENDING_FREE_COMFY_VRAM
        previous = _PENDING_FREE_COMFY_VRAM
        _PENDING_FREE_COMFY_VRAM = bool(
            kwargs.pop("Free ComfyUI VRAM Before Qwen", True)
        )
        try:
            return _PARENT_MODEL_LOADER_LOAD(self, *args, **kwargs)
        finally:
            _PENDING_FREE_COMFY_VRAM = previous

    _nodes.QwenTE模型加载器.INPUT_TYPES = _qwen_vram_input_types
    _nodes.QwenTE模型加载器.load = _qwen_vram_model_loader_load

    if _PARENT_LLAMA is not None:
        _PARENT_LLAMA_SIGNATURE = inspect.signature(_PARENT_LLAMA.__init__)

        class _QuietVerboseQwenLlama(_PARENT_LLAMA):
            def __init__(self, *args, **kwargs):
                config = getattr(_perf, "_ACTIVE_QWEN_CONFIG", None)
                verbose_logging = bool(
                    isinstance(config, dict) and config.get("_perf_verbose", False)
                )

                if verbose_logging:
                    # Keep the performance shim active, but make it believe native
                    # verbose logging is disabled. This avoids huge metadata/tensor
                    # dumps while preserving n_batch/n_ubatch/thread/FA overrides.
                    quiet_config = dict(config)
                    quiet_config["_perf_verbose"] = False
                    previous = _perf._ACTIVE_QWEN_CONFIG
                    _perf._ACTIVE_QWEN_CONFIG = quiet_config
                    try:
                        super().__init__(*args, **kwargs)
                    finally:
                        _perf._ACTIVE_QWEN_CONFIG = previous

                    flash_attn_type = _nodes._解析flash_attention类型(
                        config.get("flash_attn")
                    )
                    flash_text = (
                        "enabled"
                        if flash_attn_type == 1
                        else "disabled"
                        if flash_attn_type == 0
                        else "backend default"
                    )
                    print(
                        "[comfyUI-llama-TE verbose] Backend overrides: "
                        "n_batch=2048, n_ubatch=512, n_threads=8, "
                        f"n_threads_batch=16, flash_attn={flash_text}",
                        flush=True,
                    )
                else:
                    super().__init__(*args, **kwargs)

        # Preserve constructor feature detection performed by the upstream node.
        _QuietVerboseQwenLlama.__init__.__signature__ = _PARENT_LLAMA_SIGNATURE
        _nodes.Llama = _QuietVerboseQwenLlama

    @classmethod
    def _qwen_vram_load(cls, config: dict):
        effective_config = dict(config)

        if _PENDING_FREE_COMFY_VRAM is not None:
            free_before_load = bool(_PENDING_FREE_COMFY_VRAM)
        else:
            free_before_load = bool(
                effective_config.get("_free_comfy_vram_before_qwen", True)
            )
        effective_config["_free_comfy_vram_before_qwen"] = free_before_load

        # Anticipate the verbose flag that the performance shim will add so we
        # can accurately determine whether this is a true cache hit.
        anticipated_config = dict(effective_config)
        pending_verbose = getattr(_perf, "_PENDING_VERBOSE", None)
        if pending_verbose is not None:
            anticipated_config["_perf_verbose"] = bool(pending_verbose)

        cache_hit = bool(
            cls.model is not None
            and getattr(cls.model, "settings", None) == anticipated_config
        )
        verbose_logging = bool(anticipated_config.get("_perf_verbose", False))

        if free_before_load and not cache_hit:
            _force_free_comfy_vram(verbose_logging)
        elif verbose_logging:
            reason = "cached Qwen reuse" if cache_hit else "disabled in node"
            print(
                f"[comfyUI-llama-TE verbose] Pre-Qwen VRAM cleanup skipped: {reason}",
                flush=True,
            )

        return _PARENT_QWEN_LOAD(cls, effective_config)

    _nodes._QwenStorage.load = _qwen_vram_load
    _nodes._qwen_vram_patch_installed = True
