# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
System test for Ulysses sequence parallel backend.

This test verifies that Ulysses-SP (DeepSpeed Ulysses Sequence Parallel) works
correctly with diffusion models. It uses minimal settings to keep test time
short for CI.
"""

import gc
import os
import sys
import time
from pathlib import Path

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

models = ["riverclouds/qwen_image_random"]

PROMPT = "a photo of a cat sitting on a laptop keyboard"


def _pil_to_float_rgb_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to float32 RGB tensor in [0, 1] with shape [H, W, 3]."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


def _diff_metrics(a: Image.Image, b: Image.Image) -> tuple[float, float]:
    """Return (mean_abs_diff, max_abs_diff) over RGB pixels in [0, 1]."""
    ta = _pil_to_float_rgb_tensor(a)
    tb = _pil_to_float_rgb_tensor(b)
    assert ta.shape == tb.shape, f"Image shapes differ: {ta.shape} vs {tb.shape}"
    abs_diff = torch.abs(ta - tb)
    return abs_diff.mean().item(), abs_diff.max().item()


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


def _run_baseline(model_name: str, dtype: torch.dtype, attn_backend: str, height: int, width: int, seed: int):
    """Run baseline inference (no SP)."""
    baseline_parallel_config = DiffusionParallelConfig(ulysses_degree=1, ring_degree=1)
    baseline = Omni(
        model=model_name,
        parallel_config=baseline_parallel_config,
        dtype=dtype,
        attention_backend=attn_backend,
    )

    try:
        outputs = baseline.generate(
            PROMPT,
            OmniDiffusionSamplingParams(
                height=height,
                width=width,
                num_inference_steps=4,
                guidance_scale=0.0,
                generator=torch.Generator(current_omni_platform.device_type).manual_seed(seed),
                num_outputs_per_prompt=1,
            ),
        )
        return outputs[0].request_output[0].images
    finally:
        baseline.close()
        _cleanup_distributed()


def _run_sp(
    model_name: str,
    dtype: torch.dtype,
    attn_backend: str,
    height: int,
    width: int,
    seed: int,
    ulysses_degree: int,
    ring_degree: int,
    return_elapsed_ms: bool = False,
):
    """Run SP inference.

    Args:
        return_elapsed_ms: If True, return (images, elapsed_ms) tuple.
    """
    sp_parallel_config = DiffusionParallelConfig(ulysses_degree=ulysses_degree, ring_degree=ring_degree)
    sp = Omni(
        model=model_name,
        parallel_config=sp_parallel_config,
        dtype=dtype,
        attention_backend=attn_backend,
    )

    try:
        start_time = time.time()
        outputs = sp.generate(
            PROMPT,
            OmniDiffusionSamplingParams(
                height=height,
                width=width,
                num_inference_steps=4,
                guidance_scale=0.0,
                generator=torch.Generator(current_omni_platform.device_type).manual_seed(seed),
                num_outputs_per_prompt=1,
            ),
        )
        elapsed_ms = (time.time() - start_time) * 1000
        images = outputs[0].request_output[0].images
        if return_elapsed_ms:
            return images, elapsed_ms
        return images
    finally:
        sp.close()
        _cleanup_distributed()


@pytest.mark.parametrize("model_name", models)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("attn_backend", ["sdpa"])
def test_baseline_only(model_name: str, dtype: torch.dtype, attn_backend: str):
    """Test baseline inference only (no SP)."""
    height = 256
    width = 256
    seed = 42

    images = _run_baseline(model_name, dtype, attn_backend, height, width, seed)

    assert images is not None
    assert len(images) == 1
    assert images[0].width == width
    assert images[0].height == height


@pytest.mark.parametrize("model_name", models)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("attn_backend", ["sdpa"])
def test_sp_ulysses2_only(model_name: str, dtype: torch.dtype, attn_backend: str):
    """Test SP inference only (ulysses=2)."""
    if current_omni_platform.get_device_count() < 2:
        pytest.skip(f"Test requires 2 GPUs but only {current_omni_platform.get_device_count()} available")

    height = 256
    width = 256
    seed = 42

    images = _run_sp(model_name, dtype, attn_backend, height, width, seed, ulysses_degree=2, ring_degree=1)

    assert images is not None
    assert len(images) == 1
    assert images[0].width == width
    assert images[0].height == height


@pytest.mark.parametrize("model_name", models)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("attn_backend", ["sdpa"])
def test_sp_ring2_only(model_name: str, dtype: torch.dtype, attn_backend: str):
    """Test SP inference only (ring=2).

    This test validates that Ring attention is correctly executed by:
    1. Running Ring SP inference (ring_degree=2)
    2. Checking execution time to detect silent bypasses
    3. Comparing outputs with baseline to ensure correctness
    """
    if current_omni_platform.get_device_count() < 2:
        pytest.skip(f"Test requires 2 GPUs but only {current_omni_platform.get_device_count()} available")

    height = 256
    width = 256
    seed = 42

    # Run Ring SP inference with timing
    ring_images, ring_elapsed_ms = _run_sp(
        model_name, dtype, attn_backend, height, width, seed, ulysses_degree=1, ring_degree=2, return_elapsed_ms=True
    )

    assert ring_images is not None
    assert len(ring_images) == 1
    assert ring_images[0].width == width
    assert ring_images[0].height == height

    print(f"\n[Ring SP] Execution time: {ring_elapsed_ms:.2f}ms")

    # Execution time sanity check
    # Ring attention with 2 GPUs should take significant time (>1000ms)
    # If it's too fast (<500ms), it likely means Ring attention was bypassed
    MIN_EXPECTED_TIME_MS = 500
    if ring_elapsed_ms < MIN_EXPECTED_TIME_MS:
        print(f"[WARNING] Ring SP execution time ({ring_elapsed_ms:.2f}ms) is suspiciously fast!")
        print(f"[WARNING] Expected at least {MIN_EXPECTED_TIME_MS}ms for real Ring attention.")
        print("[WARNING] This may indicate Ring attention is not being executed correctly.")

    # Run baseline for comparison
    baseline_images = _run_baseline(model_name, dtype, attn_backend, height, width, seed)
    assert baseline_images is not None and len(baseline_images) == 1

    # Validate Ring attention output matches baseline
    mean_abs_diff, max_abs_diff = _diff_metrics(baseline_images[0], ring_images[0])

    mean_threshold = 2e-2
    max_threshold = 2e-1

    print(f"[Ring SP] Output diff vs baseline: mean={mean_abs_diff:.6e}, max={max_abs_diff:.6e}")

    assert mean_abs_diff <= mean_threshold and max_abs_diff <= max_threshold, (
        f"Ring SP output differs from baseline: mean={mean_abs_diff:.6e}, max={max_abs_diff:.6e}. "
        f"This may indicate Ring attention was not correctly executed."
    )


@pytest.mark.parametrize("model_name", models)
@pytest.mark.parametrize("ulysses_degree", [1, 2])
@pytest.mark.parametrize("ring_degree", [1, 2])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("attn_backend", ["sdpa"])
def test_sequence_parallel(
    model_name: str,
    ulysses_degree: int,
    ring_degree: int,
    dtype: torch.dtype,
    attn_backend: str,
):
    """Compare baseline (ulysses_degree=1) vs SP (ulysses_degree>1) outputs."""
    if ulysses_degree <= 1 and ring_degree <= 1:
        pytest.skip(
            "This test compares ulysses_degree * ring_degree = 1 vs ulysses_degree * ring_degree > 1; "
            "provide ulysses_degree or ring_degree>1."
        )

    sp_size = ulysses_degree * ring_degree
    if current_omni_platform.get_device_count() < sp_size:
        pytest.skip(f"Test requires {sp_size} GPUs but only {current_omni_platform.get_device_count()} available")

    height = 256
    width = 256
    seed = 42

    baseline_images = _run_baseline(model_name, dtype, attn_backend, height, width, seed)
    assert baseline_images is not None and len(baseline_images) == 1

    sp_images = _run_sp(model_name, dtype, attn_backend, height, width, seed, ulysses_degree, ring_degree)
    assert sp_images is not None and len(sp_images) == 1

    mean_abs_diff, max_abs_diff = _diff_metrics(baseline_images[0], sp_images[0])

    mean_threshold = 2e-2
    max_threshold = 2e-1

    assert mean_abs_diff <= mean_threshold and max_abs_diff <= max_threshold, (
        f"Image diff exceeded threshold: mean={mean_abs_diff:.6e}, max={max_abs_diff:.6e}"
    )


@pytest.mark.parametrize("model_name", models)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("attn_backend", ["sdpa"])
def test_sequence_parallel_ulysses4(model_name: str, dtype: torch.dtype, attn_backend: str):
    """Test SP with ulysses_degree=4."""
    ulysses_degree = 4
    ring_degree = 1

    if current_omni_platform.get_device_count() < ulysses_degree * ring_degree:
        pytest.skip(
            f"Test requires {ulysses_degree * ring_degree} GPUs but only {current_omni_platform.get_device_count()} available"
        )

    height = 272
    width = 272
    seed = 42

    baseline_images = _run_baseline(model_name, dtype, attn_backend, height, width, seed)
    assert baseline_images is not None and len(baseline_images) == 1

    sp_images = _run_sp(model_name, dtype, attn_backend, height, width, seed, ulysses_degree, ring_degree)
    assert sp_images is not None and len(sp_images) == 1

    mean_abs_diff, max_abs_diff = _diff_metrics(baseline_images[0], sp_images[0])

    mean_threshold = 2e-2
    max_threshold = 2e-1

    assert mean_abs_diff <= mean_threshold and max_abs_diff <= max_threshold, (
        f"Image diff exceeded threshold: mean={mean_abs_diff:.6e}, max={max_abs_diff:.6e}"
    )
