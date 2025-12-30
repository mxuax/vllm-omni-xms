# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
System test for Ulysses sequence parallel backend.

This test verifies that Ulysses-SP (DeepSpeed Ulysses Sequence Parallel) works
correctly with diffusion models. It uses minimal settings to keep test time
short for CI.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
from PIL import Image

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.distributed.parallel_state import device_count
from vllm_omni.diffusion.envs import get_device_name

os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "1"

# Use random weights model for testing
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


def _get_images(output):
    """Extract images from output, handling both dict and SimpleNamespace types."""
    # #region agent log
    import json

    _log_path = "/tmp/debug_get_images.log"

    def _dbg(msg, data, hyp):
        with open(_log_path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "location": "test_sequence_parallel.py:_get_images",
                        "message": msg,
                        "data": data,
                        "hypothesisId": hyp,
                        "timestamp": __import__("time").time(),
                    }
                )
                + "\n"
            )

    # #endregion

    # #region agent log - H1: Check if request_output is None
    _dbg(
        "output type and request_output",
        {
            "output_type": type(output).__name__,
            "request_output_is_none": output.request_output is None,
            "has_images_attr": hasattr(output, "images"),
        },
        "H1",
    )
    # #endregion

    # #region agent log - H3: Check if output has direct images attribute
    if hasattr(output, "images") and output.images:
        _dbg("output has direct images", {"images_count": len(output.images) if output.images else 0}, "H3")
        return output.images
    # #endregion

    # #region agent log - H1/H2: Check request_output
    if output.request_output is None:
        _dbg("request_output is None - returning None", {}, "H1")
        return None
    # #endregion

    # #region agent log - H2/H5: Check request_output type and length
    _dbg(
        "request_output info",
        {
            "type": type(output.request_output).__name__,
            "is_list": isinstance(output.request_output, list),
            "len": len(output.request_output) if hasattr(output.request_output, "__len__") else "N/A",
        },
        "H2",
    )
    # #endregion

    # #region agent log - H5: Check if empty
    if isinstance(output.request_output, list) and len(output.request_output) == 0:
        _dbg("request_output is empty list", {}, "H5")
        return None
    # #endregion

    item = output.request_output[0]

    # #region agent log - H4: Check item type and images
    _dbg(
        "item info",
        {
            "item_type": type(item).__name__,
            "is_dict": isinstance(item, dict),
            "has_images": "images" in item if isinstance(item, dict) else hasattr(item, "images"),
        },
        "H4",
    )
    # #endregion

    if isinstance(item, dict):
        # #region agent log
        _dbg(
            "returning dict images",
            {"images_value": str(item.get("images"))[:100] if item.get("images") else "None"},
            "H4",
        )
        # #endregion
        return item.get("images")
    # #region agent log
    _dbg(
        "returning attr images",
        {"images_value": str(getattr(item, "images", None))[:100] if getattr(item, "images", None) else "None"},
        "H4",
    )
    # #endregion
    return getattr(item, "images", None)


@pytest.mark.parametrize("model_name", models)
@pytest.mark.parametrize("ulysses_degree", [1, 2])
@pytest.mark.parametrize("ring_degree", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
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
            "This test compares ulysses_degree * ring_degree = 1 vs ulysses_degree * ring_degree > 1; provide ulysses_degree or ring_degree>1."
        )

    # Skip if not enough GPUs available for SP run
    if device_count() < ulysses_degree * ring_degree:
        pytest.skip(f"Test requires {ulysses_degree * ring_degree} GPUs but only {device_count()} available")

    # Use minimal settings for fast testing
    height = 256
    width = 256
    num_inference_steps = 4  # Minimal steps for fast test
    seed = 42

    # Step 1: Baseline (no Ulysses sequence parallel)
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
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0,
            generator=torch.Generator(get_device_name()).manual_seed(seed),
            num_outputs_per_prompt=1,
        )
        baseline_images = _get_images(outputs[0])
    finally:
        baseline.close()
        if dist.is_initialized():
            dist.destroy_process_group()
        for key in ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"]:
            os.environ.pop(key, None)
        time.sleep(5)  # Wait for resources to release

    assert baseline_images is not None
    assert len(baseline_images) == 1
    assert baseline_images[0].width == width
    assert baseline_images[0].height == height

    # Step 2: SP (Ulysses-SP + Ring-SP)
    sp_parallel_config = DiffusionParallelConfig(ulysses_degree=ulysses_degree, ring_degree=ring_degree)
    sp = Omni(
        model=model_name,
        parallel_config=sp_parallel_config,
        dtype=dtype,
        attention_backend=attn_backend,
    )
    try:
        outputs = sp.generate(
            PROMPT,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0,
            generator=torch.Generator(get_device_name()).manual_seed(seed),
            num_outputs_per_prompt=1,
        )
        sp_images = _get_images(outputs[0])
    finally:
        sp.close()

    assert sp_images is not None
    assert len(sp_images) == 1
    assert sp_images[0].width == width
    assert sp_images[0].height == height

    # Step 3: Compare outputs
    mean_abs_diff, max_abs_diff = _diff_metrics(baseline_images[0], sp_images[0])

    # FP16/BF16 may differ slightly due to different computation order under parallelism.
    if dtype in (torch.float16, torch.bfloat16):
        mean_threshold = 2e-2
        max_threshold = 2e-1
    else:
        mean_threshold = 1e-2
        max_threshold = 1e-1

    print(
        "Image diff stats (baseline ulysses_degree*ring_degree=1 vs SP): "
        f"mean_abs_diff={mean_abs_diff:.6e}, max_abs_diff={max_abs_diff:.6e}; "
        f"thresholds: mean<={mean_threshold:.6e}, max<={max_threshold:.6e}; "
        f"ulysses_degree={ulysses_degree}, ring_degree={ring_degree}, dtype={dtype}"
    )

    assert mean_abs_diff <= mean_threshold and max_abs_diff <= max_threshold, (
        f"Image diff exceeded threshold: mean_abs_diff={mean_abs_diff:.6e}, max_abs_diff={max_abs_diff:.6e} "
        f"(thresholds: mean<={mean_threshold:.6e}, max<={max_threshold:.6e}); "
        f"ulysses_degree={ulysses_degree}, ring_degree={ring_degree}, dtype={dtype}"
    )
