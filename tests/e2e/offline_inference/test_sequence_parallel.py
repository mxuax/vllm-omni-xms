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
    num_runs: int = 1,
) -> InferenceResult:
    """Run inference with specified configuration.

    Args:
        num_runs: Total runs including warmup. Last run is timed.
    """
    parallel_config = DiffusionParallelConfig(ulysses_degree=ulysses_degree, ring_degree=ring_degree)
    omni = Omni(
        model=model_name,
        parallel_config=parallel_config,
        dtype=dtype,
        attention_backend=attn_backend,
    )

    try:
        # Warmup runs (not timed)
        for i in range(num_runs - 1):
            _ = omni.generate(
                PROMPT,
                OmniDiffusionSamplingParams(
                    height=height,
                    width=width,
                    num_inference_steps=DEFAULT_STEPS,
                    guidance_scale=0.0,
                    generator=torch.Generator(current_omni_platform.device_type).manual_seed(seed + i + 1000),
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
# Correctness Tests
# =============================================================================


@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize(
    "ulysses_degree,ring_degree",
    [
        (2, 1),  # Ulysses only
        (1, 2),  # Ring only
        (2, 2),  # Hybrid (requires 4 GPUs)
    ],
)
def test_sp_correctness(model_name: str, ulysses_degree: int, ring_degree: int):
    """Test that SP inference produces correct outputs compared to baseline."""
    sp_size = ulysses_degree * ring_degree
    if current_omni_platform.get_device_count() < sp_size:
        pytest.skip(f"Requires {sp_size} GPUs, have {current_omni_platform.get_device_count()}")

    # Run baseline
    baseline = _run_inference(model_name, torch.bfloat16, "sdpa")
    assert len(baseline.images) == 1

    # Run SP
    sp_result = _run_inference(
        model_name,
        torch.bfloat16,
        "sdpa",
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
    )
    assert len(sp_result.images) == 1

    # Compare outputs
    mean_diff, max_diff = _diff_metrics(baseline.images[0], sp_result.images[0])
    sp_mode = (
        "ulysses"
        if ulysses_degree > 1 and ring_degree == 1
        else "ring"
        if ring_degree > 1 and ulysses_degree == 1
        else "hybrid"
    )
    print(f"\n[{sp_mode}] diff: mean={mean_diff:.6e}, max={max_diff:.6e}")

    assert mean_diff <= DIFF_MEAN_THRESHOLD and max_diff <= DIFF_MAX_THRESHOLD, (
        f"SP output differs from baseline: mean={mean_diff:.6e}, max={max_diff:.6e}"
    )


@pytest.mark.parametrize("model_name", MODELS)
def test_sp_ulysses4(model_name: str):
    """Test Ulysses SP with 4 GPUs."""
    if current_omni_platform.get_device_count() < 4:
        pytest.skip(f"Requires 4 GPUs, have {current_omni_platform.get_device_count()}")

    baseline = _run_inference(model_name, torch.bfloat16, "sdpa", height=272, width=272)
    sp_result = _run_inference(
        model_name,
        torch.bfloat16,
        "sdpa",
        ulysses_degree=4,
        ring_degree=1,
        height=272,
        width=272,
    )

    mean_diff, max_diff = _diff_metrics(baseline.images[0], sp_result.images[0])
    assert mean_diff <= DIFF_MEAN_THRESHOLD and max_diff <= DIFF_MAX_THRESHOLD


# =============================================================================
# Performance Test (with warmup)
# =============================================================================


@pytest.mark.parametrize("model_name", MODELS)
def test_sp_performance(model_name: str):
    """Performance comparison: Baseline vs Ulysses vs Ring (with warmup).

    Runs 2 iterations per config (1 warmup + 1 timed) to exclude
    NCCL initialization and JIT compilation overhead.
    """
    if current_omni_platform.get_device_count() < 2:
        pytest.skip(f"Requires 2 GPUs, have {current_omni_platform.get_device_count()}")

    num_runs = 2  # 1 warmup + 1 timed

    print("\n" + "=" * 60)
    print("SP Performance (warmup=1)")
    print("=" * 60)

    # Baseline
    baseline = _run_inference(model_name, torch.bfloat16, "sdpa", num_runs=num_runs)
    print(f"[Baseline]  1 GPU:  {baseline.elapsed_ms:.0f}ms")

    # Ulysses
    ulysses = _run_inference(
        model_name,
        torch.bfloat16,
        "sdpa",
        ulysses_degree=2,
        ring_degree=1,
        num_runs=num_runs,
    )
    ulysses_speedup = baseline.elapsed_ms / ulysses.elapsed_ms if ulysses.elapsed_ms > 0 else 0
    print(f"[Ulysses]   2 GPUs: {ulysses.elapsed_ms:.0f}ms ({ulysses_speedup:.2f}x)")

    # Ring
    ring = _run_inference(
        model_name,
        torch.bfloat16,
        "sdpa",
        ulysses_degree=1,
        ring_degree=2,
        num_runs=num_runs,
    )
    ring_speedup = baseline.elapsed_ms / ring.elapsed_ms if ring.elapsed_ms > 0 else 0
    print(f"[Ring]      2 GPUs: {ring.elapsed_ms:.0f}ms ({ring_speedup:.2f}x)")

    print("=" * 60)

    # Verify correctness
    for name, result in [("Ulysses", ulysses), ("Ring", ring)]:
        mean_diff, max_diff = _diff_metrics(baseline.images[0], result.images[0])
        print(f"[{name}] diff: mean={mean_diff:.6e}, max={max_diff:.6e}")
        assert mean_diff <= DIFF_MEAN_THRESHOLD and max_diff <= DIFF_MAX_THRESHOLD
