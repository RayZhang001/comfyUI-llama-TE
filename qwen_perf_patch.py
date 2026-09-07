# -*- coding: utf-8 -*-
"""Performance tuning and concise diagnostics for the Qwen llama.cpp backend."""

from __future__ import annotations

import inspect
import time

from . import nodes as _nodes


if not getattr(_nodes, "_qwen_perf_patch_installed", False):
    _ORIGINAL_LLAMA = _nodes.Llama
    _ORIGINAL_QWEN_LOAD = _nodes._QwenStorage.load.__func__
    _ORIGINAL_QWEN_UNLOAD = _nodes._QwenStorage.unload.__func__
    _ORIGINAL_MODEL_LOADER_INPUT_TYPES = _nodes.QwenTE模型加载器.INPUT_TYPES
    _ORIGINAL_MODEL_LOADER_LOAD = _nodes.QwenTE模型加载器.__dict__["load"]
    _ORIGINAL_INFER_RUN = _nodes.QwenTE图像推理.__dict__["run"]
    _ORIGINAL_CHAT_COMPLETION = _nodes._调用chat_completion

    _ACTIVE_QWEN_CONFIG = None
    _PENDING_VERBOSE = None
    _ACTIVE_REQUEST_VERBOSE = False

    def _fmt_bool(value) -> str:
        return "on" if bool(value) else "off"

    def _fmt_seconds(seconds: float | None) -> str:
        if seconds is None:
            return "n/a"
        return f"{seconds:.2f}s"

    def _flash_label(config: dict) -> str:
        try:
            value = _nodes._解析flash_attention类型(config.get("flash_attn"))
        except Exception:
            return "unknown"
        if value == 1:
            return "on"
        if value == 0:
            return "off"
        return "auto"

    @classmethod
    def _qwen_perf_input_types(cls):
        schema = _ORIGINAL_MODEL_LOADER_INPUT_TYPES()
        schema["required"]["Verbose Logging"] = (
            "BOOLEAN",
            {
                "default": False,
                "tooltip": (
                    "Print concise Qwen VRAM, cold-load stage, request, generation, unload, "
                    "and end-to-end timing metrics to the ComfyUI console."
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
                    kwargs.setdefault("n_batch", 2048)
                    kwargs.setdefault("n_ubatch", 512)
                    kwargs.setdefault("n_threads", 8)
                    kwargs.setdefault("n_threads_batch", 16)

                    flash_attn_type = _nodes._解析flash_attention类型(
                        config.get("flash_attn")
                    )
                    if flash_attn_type is not None:
                        kwargs["flash_attn_type"] = flash_attn_type

                # Keep native llama.cpp logging quiet. Structured diagnostics below
                # provide the metrics we need without metadata/tensor dumps.
                kwargs.pop("verbosity", None)
                super().__init__(*args, **kwargs)

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

            try:
                model = _ORIGINAL_QWEN_LOAD(cls, effective_config)
            finally:
                _ACTIVE_QWEN_CONFIG = previous

            elapsed = time.perf_counter() - started
            _nodes._qwen_diag_model_load_s = elapsed
            _nodes._qwen_diag_model_load_at = time.perf_counter()
            _nodes._qwen_diag_model_cache_hit = cache_hit

            if verbose_logging:
                load_mode = effective_config.get("_load_mode", "auto")
                lazy_mode = effective_config.get("_lazy_mode", "auto")
                no_host = _fmt_bool(effective_config.get("_no_host", False))
                model_stage = getattr(_nodes, "_qwen_diag_llama_init_s", None)
                vision_stage = getattr(_nodes, "_qwen_diag_mmproj_s", None)
                known_stage = sum(
                    value for value in (model_stage, vision_stage) if isinstance(value, (int, float))
                )
                overhead = max(0.0, elapsed - known_stage) if known_stage else None

                parts = [
                    f"total={_fmt_seconds(elapsed)}",
                    f"cached={_fmt_bool(cache_hit)}",
                    f"mode={load_mode}",
                    f"lazy={lazy_mode}",
                    f"no_host={no_host}",
                ]
                if model_stage is not None:
                    parts.append(f"model={_fmt_seconds(model_stage)}")
                if vision_stage is not None:
                    parts.append(f"vision={_fmt_seconds(vision_stage)}")
                if overhead is not None:
                    parts.append(f"other={_fmt_seconds(overhead)}")
                parts.extend(
                    [
                        f"ctx={effective_config.get('n_ctx')}",
                        f"gpu_layers={effective_config.get('n_gpu_layers')}",
                        f"FA={_flash_label(effective_config)}",
                        f"KV={effective_config.get('cache_type_k')}/{effective_config.get('cache_type_v')}",
                        f"MTP={_fmt_bool(effective_config.get('mtp_enabled'))}",
                        f"thinking={_fmt_bool(effective_config.get('think'))}",
                    ]
                )
                print("[QwenTE][LOAD] " + " | ".join(parts), flush=True)

            return model

        _nodes._QwenStorage.load = _qwen_perf_load

    @classmethod
    def _qwen_perf_unload(cls):
        settings = getattr(getattr(cls, "model", None), "settings", {}) or {}
        verbose_logging = _ACTIVE_REQUEST_VERBOSE or bool(settings.get("_perf_verbose", False))
        had_model = getattr(cls, "model", None) is not None
        started = time.perf_counter()
        try:
            return _ORIGINAL_QWEN_UNLOAD(cls)
        finally:
            if had_model:
                elapsed = time.perf_counter() - started
                _nodes._qwen_diag_unload_s = elapsed
                _nodes._qwen_diag_unload_at = time.perf_counter()
                if verbose_logging:
                    print(f"[QwenTE][UNLOAD] {_fmt_seconds(elapsed)} | llama.cpp close + cleanup", flush=True)

    _nodes._QwenStorage.unload = _qwen_perf_unload

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
            "[QwenTE][REQUEST] "
            f"images={image_count} | max_out={params.get('max_tokens')} | "
            f"temp={params.get('temperature')} | top_p={params.get('top_p')} | "
            f"top_k={params.get('top_k')} | min_p={params.get('min_p')} | "
            f"seed={params.get('seed')}",
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

        output_rate = None
        if completion_tokens is not None and elapsed > 0:
            output_rate = float(completion_tokens) / elapsed

        _nodes._qwen_diag_generation_s = elapsed
        _nodes._qwen_diag_prompt_tokens = prompt_tokens
        _nodes._qwen_diag_completion_tokens = completion_tokens
        _nodes._qwen_diag_total_tokens = total_tokens
        _nodes._qwen_diag_output_wall_tps = output_rate

        parts = [
            f"{_fmt_seconds(elapsed)}",
            f"prompt={prompt_tokens if prompt_tokens is not None else 'n/a'} tok",
            f"output={completion_tokens if completion_tokens is not None else 'n/a'} tok",
        ]
        if output_rate is not None:
            parts.append(f"output/wall={output_rate:.1f} tok/s")

        if isinstance(timings, dict) and timings:
            prompt_rate = timings.get("prompt_per_second")
            predicted_rate = timings.get("predicted_per_second")
            if prompt_rate is not None:
                parts.append(f"prefill={float(prompt_rate):.1f} tok/s")
            if predicted_rate is not None:
                parts.append(f"decode={float(predicted_rate):.1f} tok/s")

        print("[QwenTE][GEN] " + " | ".join(parts), flush=True)
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
        _nodes._qwen_diag_infer_started_at = started
        _nodes._qwen_diag_generation_s = None
        _nodes._qwen_diag_unload_s = None

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
                "[QwenTE][RUN] "
                f"mode={input_mode} | image_inputs={image_inputs} | frames={primary_frames} | "
                f"max_edge={max_edge} | max_out={max_tokens} | auto_unload={_fmt_bool(auto_unload)}",
                flush=True,
            )

        try:
            return _ORIGINAL_INFER_RUN(self, *args, **kwargs)
        finally:
            if verbose_logging:
                infer_elapsed = time.perf_counter() - started
                load_s = getattr(_nodes, "_qwen_diag_model_load_s", None)
                load_at = getattr(_nodes, "_qwen_diag_model_load_at", None)
                cleanup_s = getattr(_nodes, "_qwen_diag_vram_cleanup_s", None)
                gen_s = getattr(_nodes, "_qwen_diag_generation_s", None)
                unload_s = getattr(_nodes, "_qwen_diag_unload_s", None)

                load_in_run = isinstance(load_at, (int, float)) and load_at >= started
                recent_preload = (
                    isinstance(load_at, (int, float))
                    and load_at < started
                    and (started - load_at) <= 5.0
                )

                if load_in_run:
                    pipeline_elapsed = infer_elapsed
                elif recent_preload:
                    pipeline_elapsed = infer_elapsed
                    if isinstance(load_s, (int, float)):
                        pipeline_elapsed += load_s
                    if isinstance(cleanup_s, (int, float)):
                        pipeline_elapsed += cleanup_s
                else:
                    pipeline_elapsed = None

                summary = [f"infer={_fmt_seconds(infer_elapsed)}"]
                if pipeline_elapsed is not None:
                    summary.insert(0, f"pipeline={_fmt_seconds(pipeline_elapsed)}")
                summary.append(f"load_in_run={_fmt_bool(load_in_run)}")
                if cleanup_s is not None:
                    summary.append(f"vram={_fmt_seconds(cleanup_s)}")
                if load_s is not None:
                    summary.append(f"load={_fmt_seconds(load_s)}")
                if gen_s is not None:
                    summary.append(f"gen={_fmt_seconds(gen_s)}")
                if unload_s is not None:
                    summary.append(f"unload={_fmt_seconds(unload_s)}")
                print("[QwenTE][TOTAL] " + " | ".join(summary), flush=True)
            _ACTIVE_REQUEST_VERBOSE = previous

    _nodes.QwenTE图像推理.run = _qwen_perf_infer_run
    _nodes._qwen_perf_patch_installed = True
