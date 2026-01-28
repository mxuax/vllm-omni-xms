# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2024, Jiarui Fang.
# Adapted from https://github.com/feifeibear/long-context-attention


# test if flash_attn (FA2) is available
try:
    import flash_attn  # noqa: F401
    from flash_attn.flash_attn_interface import _flash_attn_forward  # noqa: F401

    HAS_FLASH_ATTN = True
except (ImportError, ModuleNotFoundError):
    HAS_FLASH_ATTN = False

# FA3 detection: try multiple sources (forward only, no backward needed for inference)
# Source 1: flash_attn_interface (from flash-attention source build for Hopper)
# Source 2: fa3_fwd_interface (from fa3-fwd PyPI package)
HAS_FLASH_ATTN_HOPPER = False
flash_attn_forward_hopper = None
flash3_attn_func = None

try:
    from flash_attn_interface import _flash_attn_forward as flash_attn_forward_hopper  # noqa: F401
    from flash_attn_interface import flash_attn_func as flash3_attn_func  # noqa: F401

    HAS_FLASH_ATTN_HOPPER = True
except (ImportError, ModuleNotFoundError):
    pass

# Fallback: try fa3_fwd_interface (PyPI package, built from flash-attention hopper)
if not HAS_FLASH_ATTN_HOPPER:
    try:
        from fa3_fwd_interface import _flash_attn_forward as flash_attn_forward_hopper  # noqa: F401
        from fa3_fwd_interface import flash_attn_func as flash3_attn_func  # noqa: F401

        HAS_FLASH_ATTN_HOPPER = True
    except (ImportError, ModuleNotFoundError):
        pass

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
