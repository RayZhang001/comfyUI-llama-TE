# -*- coding: utf-8 -*-
"""Experimental Qwen model-loading controls for cold-load benchmarking."""

from __future__ import annotations

import inspect

from . import nodes as _nodes
from . import qwen_perf_patch as _perf

try:
    import llama_cpp.llama_cpp as _llama_low
except Exception:
    _llama_low = None


if not getattr(_nodes, "_qwen_loader_tuning_installed", False):
    _PARENT_LLAMA = _nodes.Llama
    _PARENT_QWEN_LOAD = _nodes._QwenStorage.load.__func__
    _PARENT_MODEL_LOADER_INPUT_TYPES = _nodes.QwenTE模型加载器.INPUT_TYPES
    _PARENT_MODEL_LOADER_LOAD = _nodes.QwenTE模型加载器.__dict__["load"]

    _PENDING_LOAD_MODE = None
    _PENDING_LAZY_MODE = None
    _PENDING_NO_HOST = None

    _LOAD_MODE_NAMES = {
        "auto": "LLAMA_LOAD_MODE_AUTO",
        "mmap": "LLAMA_LOAD_MODE_MMAP",
        "mmap+mlock": "LLAMA_LOAD_MODE_MMAP_MLOCK",
        "mlock": "LLAMA_LOAD_MODE_MLOCK",
        "none": "LLAMA_LOAD_MODE_NONE",
        "dio": "LLAMA_LOAD_MODE_DIRECT_IO",
    }

    _LAZY_MODE_NAMES = {
        "auto": "LLAMA_LAZY_MODE_AUTO",
        "off": "LLAMA_LAZY_MODE_OFF",
        "on": "LLAMA_LAZY_MODE_ON",
    }

    def _resolve_enum(enum_name: str, member_name: str):
        if _llama_low is None:
            raise RuntimeError("llama-cpp-python low-level bindings are unavailable.")
        enum_type = getattr(_llama_low, enum_name, None)
        if enum_type is None:
            raise RuntimeError(
                f"The installed llama-cpp-python build does not expose {enum_name}."
            )
        value = getattr(enum_type, member_name, None)
        if value is None:
            raise RuntimeError(
                f"The installed llama-cpp-python build does not expose {member_name}."
            )
        return value

    @classmethod
    def _qwen_loader_tuning_input_types(cls):
        schema = _PARENT_MODEL_LOADER_INPUT_TYPES()
        schema["required"]["Model Load Mode"] = (
            ["auto", "mmap", "mmap+mlock", "mlock", "none", "dio"],
            {
                "default": "mmap",
                "tooltip": (
                    "Cold-load strategy passed to llama.cpp. mmap is the recommended baseline. "
                    "mmap+mlock is worth testing on high-RAM systems. none and dio are diagnostic "
                    "alternatives and may be slower or unsupported on some systems."
                ),
            },
        )
        schema["required"]["Lazy Tensor Mode"] = (
            ["auto", "off", "on"],
            {
                "default": "auto",
                "tooltip": (
                    "Controls llama.cpp on-demand tensor reads. Dense Qwen3.8-27B is expected "
                    "to show little difference, but the option is exposed for measurement."
                ),
            },
        )
        schema["required"]["Experimental No Host Buffers"] = (
            "BOOLEAN",
            {
                "default": False,
                "tooltip": (
                    "Pass no_host=True to llama.cpp. This may reduce host-buffer work on full "
                    "GPU offload, but it is experimental and should be A/B tested."
                ),
            },
        )
        return schema

    def _qwen_loader_tuning_model_loader_load(self, *args, **kwargs):
        global _PENDING_LOAD_MODE, _PENDING_LAZY_MODE, _PENDING_NO_HOST

        previous_load = _PENDING_LOAD_MODE
        previous_lazy = _PENDING_LAZY_MODE
        previous_no_host = _PENDING_NO_HOST

        _PENDING_LOAD_MODE = str(kwargs.pop("Model Load Mode", "mmap"))
        _PENDING_LAZY_MODE = str(kwargs.pop("Lazy Tensor Mode", "auto"))
        _PENDING_NO_HOST = bool(kwargs.pop("Experimental No Host Buffers", False))

        try:
            return _PARENT_MODEL_LOADER_LOAD(self, *args, **kwargs)
        finally:
            _PENDING_LOAD_MODE = previous_load
            _PENDING_LAZY_MODE = previous_lazy
            _PENDING_NO_HOST = previous_no_host

    _nodes.QwenTE模型加载器.INPUT_TYPES = _qwen_loader_tuning_input_types
    _nodes.QwenTE模型加载器.load = _qwen_loader_tuning_model_loader_load

    if _PARENT_LLAMA is not None:
        _PARENT_LLAMA_SIGNATURE = inspect.signature(_PARENT_LLAMA.__init__)

        class _QwenLoaderTunedLlama(_PARENT_LLAMA):
            def __init__(self, *args, **kwargs):
                config = getattr(_perf, "_ACTIVE_QWEN_CONFIG", None)
                if isinstance(config, dict):
                    load_mode_name = str(config.get("_load_mode", "mmap"))
                    lazy_mode_name = str(config.get("_lazy_mode", "auto"))
                    no_host = bool(config.get("_no_host", False))

                    if load_mode_name not in _LOAD_MODE_NAMES:
                        raise ValueError(f"Unknown Qwen model load mode: {load_mode_name}")
                    if lazy_mode_name not in _LAZY_MODE_NAMES:
                        raise ValueError(f"Unknown Qwen lazy tensor mode: {lazy_mode_name}")

                    kwargs["load_mode"] = _resolve_enum(
                        "llama_load_mode", _LOAD_MODE_NAMES[load_mode_name]
                    )
                    kwargs["lazy_mode"] = _resolve_enum(
                        "llama_lazy_mode", _LAZY_MODE_NAMES[lazy_mode_name]
                    )
                    kwargs["no_host"] = no_host

                super().__init__(*args, **kwargs)

        _QwenLoaderTunedLlama.__init__.__signature__ = _PARENT_LLAMA_SIGNATURE
        _nodes.Llama = _QwenLoaderTunedLlama

    @classmethod
    def _qwen_loader_tuned_load(cls, config: dict):
        effective_config = dict(config)

        if _PENDING_LOAD_MODE is not None:
            effective_config["_load_mode"] = _PENDING_LOAD_MODE
        else:
            effective_config.setdefault("_load_mode", "mmap")

        if _PENDING_LAZY_MODE is not None:
            effective_config["_lazy_mode"] = _PENDING_LAZY_MODE
        else:
            effective_config.setdefault("_lazy_mode", "auto")

        if _PENDING_NO_HOST is not None:
            effective_config["_no_host"] = bool(_PENDING_NO_HOST)
        else:
            effective_config.setdefault("_no_host", False)

        return _PARENT_QWEN_LOAD(cls, effective_config)

    _nodes._QwenStorage.load = _qwen_loader_tuned_load
    _nodes._qwen_loader_tuning_installed = True
