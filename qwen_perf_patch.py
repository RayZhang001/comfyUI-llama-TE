# -*- coding: utf-8 -*-
"""Local performance tuning for the Qwen llama.cpp backend.

This keeps the upstream node implementation intact while making Qwen model
construction match the known-good standalone llama.cpp settings used on the
RTX 3090 test system:

- n_batch=2048
- n_ubatch=512
- n_threads=8
- n_threads_batch=16
- honor the existing Qwen Flash Attention selector

Vision/MTP behavior is intentionally unchanged. In particular, the upstream
node still disables its Python speculative path when an mmproj is loaded.
"""

from __future__ import annotations

import inspect

from . import nodes as _nodes


if not getattr(_nodes, "_qwen_perf_patch_installed", False):
    _ORIGINAL_LLAMA = _nodes.Llama
    _ORIGINAL_QWEN_LOAD = _nodes._QwenStorage.load.__func__
    _ACTIVE_QWEN_CONFIG = None

    if _ORIGINAL_LLAMA is not None:
        _ORIGINAL_LLAMA_INIT_SIGNATURE = inspect.signature(_ORIGINAL_LLAMA.__init__)

        class _QwenPerfLlama(_ORIGINAL_LLAMA):
            def __init__(self, *args, **kwargs):
                config = _ACTIVE_QWEN_CONFIG
                if config is not None:
                    # Pin the same execution parameters as the known-good
                    # standalone llama-server command instead of relying on
                    # wrapper defaults that can change between wheels.
                    kwargs.setdefault("n_batch", 2048)
                    kwargs.setdefault("n_ubatch", 512)
                    kwargs.setdefault("n_threads", 8)
                    kwargs.setdefault("n_threads_batch", 16)

                    # The upstream Qwen loader stores the UI choice in config
                    # but never forwards it to Llama(...). Preserve the UI's
                    # three-state behavior: only explicit enable/disable
                    # overrides llama.cpp; "不开启" leaves the backend default.
                    flash_attn_type = _nodes._解析flash_attention类型(
                        config.get("flash_attn", "不开启")
                    )
                    if flash_attn_type is not None:
                        kwargs["flash_attn_type"] = flash_attn_type

                    flash_text = (
                        "enabled"
                        if flash_attn_type == 1
                        else "disabled"
                        if flash_attn_type == 0
                        else "backend default"
                    )
                    print(
                        "[comfyUI-llama-TE perf] Qwen backend overrides: "
                        "n_batch=2048, n_ubatch=512, n_threads=8, "
                        f"n_threads_batch=16, flash_attn={flash_text}",
                        flush=True,
                    )

                super().__init__(*args, **kwargs)

        # nodes._llama构造参数是否可用() introspects Llama.__init__.
        # Preserve the original signature so feature checks such as
        # speculative/type_k/type_v/ctx_checkpoints continue to work.
        _QwenPerfLlama.__init__.__signature__ = _ORIGINAL_LLAMA_INIT_SIGNATURE
        _nodes.Llama = _QwenPerfLlama

        @classmethod
        def _qwen_perf_load(cls, config: dict):
            global _ACTIVE_QWEN_CONFIG
            previous = _ACTIVE_QWEN_CONFIG
            _ACTIVE_QWEN_CONFIG = config
            try:
                return _ORIGINAL_QWEN_LOAD(cls, config)
            finally:
                _ACTIVE_QWEN_CONFIG = previous

        _nodes._QwenStorage.load = _qwen_perf_load

    _nodes._qwen_perf_patch_installed = True
