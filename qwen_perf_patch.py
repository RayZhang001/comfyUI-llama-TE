# -*- coding: utf-8 -*-
"""Local performance tuning and diagnostics for the Qwen llama.cpp backend.

This keeps the upstream node implementation intact while making Qwen model
construction match the known-good standalone llama.cpp settings used on the
RTX 3090 test system:

- n_batch=2048
- n_ubatch=512
- n_threads=8
- n_threads_batch=16
- honor the existing Qwen Flash Attention selector

It also adds an English-only "Verbose Logging" toggle to the Qwen model loader.
When enabled, important backend settings and request metrics are printed to the
ComfyUI console. Native llama.cpp info logging is also enabled for exact prompt
and generation timing when supported by the installed llama-cpp-python build.

Vision/MTP behavior is intentionally unchanged. In particular, the upstream
node still disables its Python speculative path when an mmproj is loaded.
"""

from __future__ import annotations

import inspect
import os
import time

from . import nodes as _nodes


if not getattr(_nodes, "_qwen_perf_patch_installed", False):
    _ORIGINAL_LLAMA = _nodes.Llama
    _ORIGINAL_QWEN_LOAD = _nodes._QwenStorage.load.__func__
    _ORIGINAL_MODEL_LOADER_INPUT_TYPES = _nodes.QwenTE模型加载器.INPUT_TYPES
    _ORIGINAL_MODEL_LOADER_LOAD = _nodes.QwenTE模型加载器.__dict__["load"]
    _ORIGINAL_INFER_RUN = _nodes.QwenTE图像推理.__dict__["run"]
    _ORIGINAL_CHAT_COMPLETION = _nodes._调用chat_completion

    _ACTIVE_QWEN_CONFIG = None
    _PENDING_VERBOSE = None
    _ACTIVE_REQUEST_VERBOSE = False

    @classmethod
    def _qwen_perf_input_types(cls):
        schema = _ORIGINAL_MODEL_LOADER_INPUT_TYPES()
        schema["required"]["Verbose Logging"] = (
            "BOOLEAN",
            {
                "default": False,
                "tooltip": (
                    "Print effective Qwen backend settings and per-request timing/token "
                    "metrics to the ComfyUI console. Also enables llama.cpp info-level logs."
                ),
            },
        )
        return schema

    def _qwen_perf_model_loader_load(self, *args, **kwargs):
        global _PENDING_VERBOSE
        previous = _PENDING_VERBOSE
        _PENDING_VERBOSE = bool(kwargs.pop("Verbose Logging", False))
        try:
            return _ORIGINAL_MODEL_LOADER_LOAD(self, *args, **kwargs)
        finally:
            _PENDING_VERBOSE = previous

    _nodes.QwenTE模型加载器.INPUT_TYPES = _qwen_perf_input_types
    _nodes.QwenTE模型加载器.load = _qwen_perf_model_loader_load

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
                    # existing three-state behavior without adding new labels.
                    flash_attn_type = _nodes._解析flash_attention类型(
                        config.get("flash_attn")
                    )
                    if flash_attn_type is not None:
                        kwargs["flash_attn_type"] = flash_attn_type

                    verbose_logging = bool(config.get("_perf_verbose", False))
                    if verbose_logging:
                        # In the current JamePeng wrapper, verbosity takes
                        # precedence over verbose=False and exposes native
                        # llama.cpp prompt/eval timing lines.
                        kwargs["verbosity"] = 3

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

                super().__init__(*args, **kwargs)

        # Feature checks in the upstream node introspect Llama.__init__.
        # Preserve the original constructor signature so checks for speculative,
        # KV cache types, checkpoints, and other optional parameters still work.
        _QwenPerfLlama.__init__.__signature__ = _ORIGINAL_LLAMA_INIT_SIGNATURE
        _nodes.Llama = _QwenPerfLlama

        @classmethod
        def _qwen_perf_load(cls, config: dict):
            global _ACTIVE_QWEN_CONFIG

            effective_config = dict(config)
            if _PENDING_VERBOSE is not None:
                effective_config["_perf_verbose"] = bool(_PENDING_VERBOSE)

            verbose_logging = bool(effective_config.get("_perf_verbose", False))
            cache_hit = bool(cls.model and cls.model.settings == effective_config)
            previous = _ACTIVE_QWEN_CONFIG
            _ACTIVE_QWEN_CONFIG = effective_config
            started = time.perf_counter()

            if verbose_logging:
                flash_mode = effective_config.get("flash_attn")
                print(
                    "[comfyUI-llama-TE verbose] Model config: "
                    f"family={effective_config.get('family')}, "
                    f"model={effective_config.get('model')}, "
                    f"mmproj={effective_config.get('mmproj')}, "
                    f"n_ctx={effective_config.get('n_ctx')}, "
                    f"n_gpu_layers={effective_config.get('n_gpu_layers')}, "
                    f"kv_k={effective_config.get('cache_type_k')}, "
                    f"kv_v={effective_config.get('cache_type_v')}, "
                    f"flash_attn_ui={flash_mode}, "
                    f"mtp_requested={effective_config.get('mtp_enabled')}, "
                    f"mtp_draft_tokens={effective_config.get('mtp_draft_tokens')}, "
                    f"thinking={effective_config.get('think')}, "
                    f"reasoning_effort={effective_config.get('reasoning_effort')}, "
                    f"cache_hit={cache_hit}",
                    flush=True,
                )

            try:
                model = _ORIGINAL_QWEN_LOAD(cls, effective_config)
            finally:
                _ACTIVE_QWEN_CONFIG = previous

            if verbose_logging:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                print(
                    "[comfyUI-llama-TE verbose] Model load: "
                    f"wall_ms={elapsed_ms:.2f}, reused_cached_model={cache_hit}",
                    flush=True,
                )

            return model

        _nodes._QwenStorage.load = _qwen_perf_load

    def _count_message_images(messages) -> int:
        count = 0
        for message in messages or []:
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    count += 1
        return count

    def _qwen_perf_chat_completion(llm, *, messages, params: dict):
        if not _ACTIVE_REQUEST_VERBOSE:
            return _ORIGINAL_CHAT_COMPLETION(llm, messages=messages, params=params)

        image_count = _count_message_images(messages)
        print(
            "[comfyUI-llama-TE verbose] Completion request: "
            f"images={image_count}, max_tokens={params.get('max_tokens')}, "
            f"temperature={params.get('temperature')}, top_p={params.get('top_p')}, "
            f"top_k={params.get('top_k')}, min_p={params.get('min_p')}, "
            f"repeat_penalty={params.get('repeat_penalty')}, "
            f"frequency_penalty={params.get('frequency_penalty')}, "
            f"presence_penalty={params.get('presence_penalty')}, seed={params.get('seed')}",
            flush=True,
        )

        started = time.perf_counter()
        result = _ORIGINAL_CHAT_COMPLETION(llm, messages=messages, params=params)
        elapsed = time.perf_counter() - started

        usage = result.get("usage", {}) if isinstance(result, dict) else {}
        timings = result.get("timings", {}) if isinstance(result, dict) else {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")

        if completion_tokens is not None and elapsed > 0:
            output_per_wall_second = float(completion_tokens) / elapsed
            output_rate_text = f"{output_per_wall_second:.2f}"
        else:
            output_rate_text = "n/a"

        metric_parts = [
            f"wall_ms={elapsed * 1000.0:.2f}",
            f"prompt_tokens={prompt_tokens if prompt_tokens is not None else 'n/a'}",
            f"completion_tokens={completion_tokens if completion_tokens is not None else 'n/a'}",
            f"total_tokens={total_tokens if total_tokens is not None else 'n/a'}",
            f"completion_tokens_per_wall_s={output_rate_text}",
        ]

        if isinstance(timings, dict) and timings:
            for key in (
                "prompt_ms",
                "prompt_per_second",
                "predicted_ms",
                "predicted_per_second",
            ):
                if key in timings:
                    metric_parts.append(f"{key}={timings[key]}")

        print(
            "[comfyUI-llama-TE verbose] Completion metrics: " + ", ".join(metric_parts),
            flush=True,
        )
        return result

    _nodes._调用chat_completion = _qwen_perf_chat_completion

    _INFER_SIGNATURE = inspect.signature(_ORIGINAL_INFER_RUN)
    _INFER_PARAMETER_NAMES = list(_INFER_SIGNATURE.parameters)

    def _bound_value(bound, position: int, default=None):
        if position >= len(_INFER_PARAMETER_NAMES):
            return default
        return bound.arguments.get(_INFER_PARAMETER_NAMES[position], default)

    def _qwen_perf_infer_run(self, *args, **kwargs):
        global _ACTIVE_REQUEST_VERBOSE

        try:
            bound = _INFER_SIGNATURE.bind_partial(self, *args, **kwargs)
        except Exception:
            bound = None

        model = _bound_value(bound, 1) if bound is not None else (args[0] if args else None)
        settings = getattr(model, "settings", {}) or {}
        verbose_logging = bool(settings.get("_perf_verbose", False))

        previous = _ACTIVE_REQUEST_VERBOSE
        _ACTIVE_REQUEST_VERBOSE = verbose_logging
        started = time.perf_counter()

        if verbose_logging and bound is not None:
            input_mode = _bound_value(bound, 2)
            max_edge = _bound_value(bound, 6)
            max_tokens = _bound_value(bound, 7)
            auto_unload = _bound_value(bound, 16, False)

            image_inputs = 0
            primary_frames = 0
            for position in range(17, min(25, len(_INFER_PARAMETER_NAMES))):
                image = _bound_value(bound, position)
                if image is None:
                    continue
                image_inputs += 1
                if position == 17 and hasattr(image, "shape"):
                    try:
                        primary_frames = int(image.shape[0])
                    except Exception:
                        primary_frames = 0

            print(
                "[comfyUI-llama-TE verbose] Inference start: "
                f"input_mode={input_mode}, image_inputs={image_inputs}, "
                f"primary_frames={primary_frames}, max_edge={max_edge}, "
                f"max_tokens={max_tokens}, auto_unload={bool(auto_unload)}",
                flush=True,
            )

        try:
            return _ORIGINAL_INFER_RUN(self, *args, **kwargs)
        finally:
            if verbose_logging:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                print(
                    f"[comfyUI-llama-TE verbose] Inference end: wall_ms={elapsed_ms:.2f}",
                    flush=True,
                )
            _ACTIVE_REQUEST_VERBOSE = previous

    _nodes.QwenTE图像推理.run = _qwen_perf_infer_run
    _nodes._qwen_perf_patch_installed = True
