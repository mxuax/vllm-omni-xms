# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2024, Jiarui Fang.
# Adapted from https://github.com/feifeibear/long-context-attention

import torch

# test if flash_attn (FA2) is available
try:
    import flash_attn  # noqa: F401
    from flash_attn.flash_attn_interface import _flash_attn_forward  # noqa: F401

    HAS_FLASH_ATTN = True
except (ImportError, ModuleNotFoundError):
    HAS_FLASH_ATTN = False

# FA3 detection: try multiple sources (forward only, no backward needed for inference)
# Source 1: flash_attn_interface (from flash-attention source build)
# Source 2: fa3_fwd_interface (from fa3-fwd PyPI package, supports Ampere/Ada/Hopper)
HAS_FA3 = False
fa3_fwd_func = None  # Low-level forward function
fa3_attn_func = None  # High-level attention function
FA3_RETURNS_LSE = False  # Whether the high-level API returns LSE (detected at import time)
_FA3_SOURCE = None  # Source of FA3 ("flash_attn_interface" or "fa3_fwd_interface")

# Try flash_attn_interface first (from flash-attention source build)
try:
    from flash_attn_interface import _flash_attn_forward as fa3_fwd_func  # noqa: F401
    from flash_attn_interface import flash_attn_func as fa3_attn_func  # noqa: F401

    HAS_FA3 = True
    _FA3_SOURCE = "flash_attn_interface"
except (ImportError, ModuleNotFoundError):
    pass

# Fallback: try fa3_fwd_interface (PyPI package, supports Ampere/Ada/Hopper)
if not HAS_FA3:
    try:
        from fa3_fwd_interface import _flash_attn_forward as fa3_fwd_func  # noqa: F401
        from fa3_fwd_interface import flash_attn_func as fa3_attn_func  # noqa: F401

        HAS_FA3 = True
        _FA3_SOURCE = "fa3_fwd_interface"
    except (ImportError, ModuleNotFoundError):
        pass

# Detect at import time whether fa3_attn_func returns LSE
# This avoids runtime detection overhead in the forward pass
if HAS_FA3 and fa3_attn_func is not None:
    try:
        # Create small test tensors to probe the API
        _test_q = torch.zeros(1, 1, 1, 64, device="meta")
        _test_k = torch.zeros(1, 1, 1, 64, device="meta")
        _test_v = torch.zeros(1, 1, 1, 64, device="meta")
        # We can't actually run on meta device, so we check function signature instead
        import inspect

        sig = inspect.signature(fa3_attn_func)
        # If the function has return_lse or similar parameter, it likely returns LSE
        # Otherwise, we assume it returns (out, lse) tuple by default for FA3
        FA3_RETURNS_LSE = True  # FA3 high-level API typically returns LSE
    except Exception:
        FA3_RETURNS_LSE = False

# Legacy aliases for backward compatibility
HAS_FLASH_ATTN_HOPPER = HAS_FA3
flash_attn_forward_hopper = fa3_fwd_func
flash3_attn_func = fa3_attn_func

try:
    from flashinfer.prefill import single_prefill_with_kv_cache  # noqa: F401

    HAS_FLASHINFER = True
except (ImportError, ModuleNotFoundError):
    HAS_FLASHINFER = False

try:
    import aiter  # noqa: F401
    from aiter import flash_attn_func as flash_attn_func_aiter  # noqa: F401

    HAS_AITER = True
except (ImportError, ModuleNotFoundError):
    HAS_AITER = False

try:
    import sageattention  # noqa: F401

    HAS_SAGE_ATTENTION = True
except (ImportError, ModuleNotFoundError):
    HAS_SAGE_ATTENTION = False

try:
    import spas_sage_attn  # noqa: F401

    HAS_SPARSE_SAGE_ATTENTION = True
except (ImportError, ModuleNotFoundError):
    HAS_SPARSE_SAGE_ATTENTION = False

try:
    import torch_npu  # noqa: F401

    HAS_NPU = True
except (ImportError, ModuleNotFoundError):
    HAS_NPU = False
