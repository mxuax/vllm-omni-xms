# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
System test for Sequence Parallel (SP) backends: Ulysses and Ring attention.

Tests verify that SP inference produces correct outputs compared to baseline.
"""

import gc
import os
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
import torch
import torch.distributed as dist
from PIL import Image

from vllm_omni.inputs.data import OmniDiffusionSamplingParams

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.platforms import current_omni_platform

# Test configuration
MODELS = ["riverclouds/qwen_image_random"]
PROMPT = "a photo of a cat sitting on a laptop keyboard"
DEFAULT_HEIGHT = 256
DEFAULT_WIDTH = 256
DEFAULT_SEED = 42
DEFAULT_STEPS = 4
DIFF_MEAN_THRESHOLD = 2e-2
DIFF_MAX_THRESHOLD = 2e-1


class InferenceResult(NamedTuple):
    """Result of an inference run."""

    images: list[Image.Image]
    elapsed_ms: float


def _cleanup_distributed():
    """Clean up distributed environment and GPU resources."""
    if dist.is_initialized():
        dist.destroy_process_group()

    for key in ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"]:
        os.environ.pop(key, None)

    gc.collect()
    if current_omni_platform.is_available():
        current_omni_platform.empty_cache()
        current_omni_platform.synchronize()

    time.sleep(5)


def _diff_metrics(a: Image.Image, b: Image.Image) -> tuple[float, float]:
    """Return (mean_abs_diff, max_abs_diff) over RGB pixels in [0, 1]."""
    ta = torch.from_numpy(np.asarray(a.convert("RGB"), dtype=np.float32) / 255.0)
    tb = torch.from_numpy(np.asarray(b.convert("RGB"), dtype=np.float32) / 255.0)
    assert ta.shape == tb.shape, f"Image shapes differ: {ta.shape} vs {tb.shape}"
    abs_diff = torch.abs(ta - tb)
    return abs_diff.mean().item(), abs_diff.max().item()


def _run_inference(
    model_name: str,
    dtype: torch.dtype,
    attn_backend: str,
    ulysses_degree: int = 1,
    ring_degree: int = 1,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    seed: int = DEFAULT_SEED,
    warmup: bool = True,
) -> InferenceResult:
    """Run inference with specified configuration.

    Args:
        warmup: If True, run one warmup iteration before the timed run.
    """
    parallel_config = DiffusionParallelConfig(ulysses_degree=ulysses_degree, ring_degree=ring_degree)
    omni = Omni(
        model=model_name,
        parallel_config=parallel_config,
        dtype=dtype,
        attention_backend=attn_backend,
    )

    try:
        # Warmup run (not timed)
        if warmup:
            _ = omni.generate(
                PROMPT,
                OmniDiffusionSamplingParams(
                    height=height,
                    width=width,
                    num_inference_steps=DEFAULT_STEPS,
                    guidance_scale=0.0,
                    generator=torch.Generator(current_omni_platform.device_type).manual_seed(seed + 1000),
                    num_outputs_per_prompt=1,
                ),
            )

        # Timed run
        start = time.time()
        outputs = omni.generate(
            PROMPT,
            OmniDiffusionSamplingParams(
                height=height,
                width=width,
                num_inference_steps=DEFAULT_STEPS,
                guidance_scale=0.0,
                generator=torch.Generator(current_omni_platform.device_type).manual_seed(seed),
                num_outputs_per_prompt=1,
            ),
        )
        elapsed_ms = (time.time() - start) * 1000

        return InferenceResult(
            images=outputs[0].request_output[0].images,
            elapsed_ms=elapsed_ms,
        )
    finally:
        omni.close()
        _cleanup_distributed()


# =============================================================================
# Correctness & Performance Tests
# =============================================================================

# SP configurations: (ulysses_degree, ring_degree, height, width)
SP_CONFIGS = [
    (2, 1, DEFAULT_HEIGHT, DEFAULT_WIDTH),  # Ulysses (2 GPUs)
    (1, 2, DEFAULT_HEIGHT, DEFAULT_WIDTH),  # Ring (2 GPUs)
    (2, 2, DEFAULT_HEIGHT, DEFAULT_WIDTH),  # Hybrid (4 GPUs)
    (4, 1, 272, 272),  # Ulysses-4 (4 GPUs, non-standard shape)
]


def _get_sp_mode(ulysses_degree: int, ring_degree: int) -> str:
    """Get SP mode name for logging."""
    if ulysses_degree > 1 and ring_degree == 1:
        return f"ulysses-{ulysses_degree}"
    elif ring_degree > 1 and ulysses_degree == 1:
        return f"ring-{ring_degree}"
    else:
        return f"hybrid-{ulysses_degree}x{ring_degree}"


@pytest.mark.parametrize("model_name", MODELS)
def test_sp_correctness(model_name: str):
    """Test that SP inference produces correct outputs and measure performance.

    Runs baseline once per unique (height, width), then tests all SP configs.
    """
    device_count = current_omni_platform.get_device_count()

    # Cache baseline results by (height, width)
    baseline_cache: dict[tuple[int, int], InferenceResult] = {}

    # Collect results for summary
    results: list[tuple[str, int, float, float, float, float]] = []

    for ulysses_degree, ring_degree, height, width in SP_CONFIGS:
        sp_size = ulysses_degree * ring_degree
        sp_mode = _get_sp_mode(ulysses_degree, ring_degree)

        if device_count < sp_size:
            print(f"\n[{sp_mode}] SKIPPED (requires {sp_size} GPUs, have {device_count})")
            continue

        # Get or compute baseline for this (height, width)
        cache_key = (height, width)
        if cache_key not in baseline_cache:
            baseline = _run_inference(model_name, torch.bfloat16, "sdpa", height=height, width=width)
            assert len(baseline.images) == 1
            baseline_cache[cache_key] = baseline
            print(f"\n[baseline] {height}x{width}: {baseline.elapsed_ms:.0f}ms")
        else:
            baseline = baseline_cache[cache_key]

        # Run SP
        sp_result = _run_inference(
            model_name,
            torch.bfloat16,
            "sdpa",
            ulysses_degree=ulysses_degree,
            ring_degree=ring_degree,
            height=height,
            width=width,
        )
        assert len(sp_result.images) == 1

        # Compare outputs (correctness)
        mean_diff, max_diff = _diff_metrics(baseline.images[0], sp_result.images[0])

        # Performance metrics
        speedup = baseline.elapsed_ms / sp_result.elapsed_ms if sp_result.elapsed_ms > 0 else 0

        print(
            f"[{sp_mode}] {sp_size} GPUs | "
            f"sp: {sp_result.elapsed_ms:.0f}ms, speedup: {speedup:.2f}x | "
            f"diff: mean={mean_diff:.6e}, max={max_diff:.6e}"
        )

        # Store results for final assertion
        results.append((sp_mode, sp_size, speedup, mean_diff, max_diff, sp_result.elapsed_ms))

        # Assert correctness
        assert mean_diff <= DIFF_MEAN_THRESHOLD and max_diff <= DIFF_MAX_THRESHOLD, (
            f"[{sp_mode}] SP output differs from baseline: mean={mean_diff:.6e}, max={max_diff:.6e}"
        )

    # Summary
    if results:
        print("\n" + "=" * 60)
        print("Summary:")
        for sp_mode, sp_size, speedup, mean_diff, max_diff, elapsed_ms in results:
            print(f"  [{sp_mode}] {sp_size} GPUs: {elapsed_ms:.0f}ms ({speedup:.2f}x)")
        print("=" * 60)
