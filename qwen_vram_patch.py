# -*- coding: utf-8 -*-
"""VRAM coordination for the patched Qwen llama.cpp backend."""

from __future__ import annotations

import gc
import time

try:
    import torch
except Exception:
    torch = None

from . import nodes as _nodes
from . import qwen_perf_patch as _perf


if not getattr(_nodes, "_qwen_vram_patch_installed", False):
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

    def _force_free_comfy_vram(verbose_logging: bool) -> None:
        before = _gpu_mem_mib()
        started = time.perf_counter()

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
        elapsed = time.perf_counter() - started

        _nodes._qwen_diag_vram_cleanup_s = elapsed
        _nodes._qwen_diag_vram_before_mib = before[0] if before is not None else None
        _nodes._qwen_diag_vram_after_mib = after[0] if after is not None else None

        if verbose_logging:
            if before is not None and after is not None:
                reclaimed = after[0] - before[0]
                print(
                    "[QwenTE][VRAM] "
                    f"{elapsed:.2f}s | free {before[0] / 1024.0:.1f} -> {after[0] / 1024.0:.1f} GiB "
                    f"| reclaimed {reclaimed / 1024.0:.1f} GiB",
                    flush=True,
                )
            else:
                print(
                    f"[QwenTE][VRAM] {elapsed:.2f}s | memory counters unavailable",
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
                    "and clear CUDA caches so llama.cpp starts with maximum free VRAM."
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
        else:
            _nodes._qwen_diag_vram_cleanup_s = 0.0
            if verbose_logging:
                reason = "cached Qwen reuse" if cache_hit else "disabled"
                print(f"[QwenTE][VRAM] skipped | {reason}", flush=True)

        return _PARENT_QWEN_LOAD(cls, effective_config)

    _nodes._QwenStorage.load = _qwen_vram_load
    _nodes._qwen_vram_patch_installed = True
