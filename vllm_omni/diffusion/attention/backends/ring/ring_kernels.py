# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Jiarui Fang.
# Adapted from https://github.com/feifeibear/long-context-attention

import math

import torch

from .ring_globals import (
    HAS_AITER,
    HAS_FLASH_ATTN,
    HAS_FLASH_ATTN_HOPPER,
    HAS_FLASHINFER,
    HAS_NPU,
    flash3_attn_func,
    flash_attn_forward_hopper,
)

_scaled_dot_product_flash_attention = torch.ops.aten._scaled_dot_product_flash_attention
_scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention

try:
    import torch_musa  # noqa: F401

    _scaled_dot_product_flash_attention = torch.ops.aten._scaled_dot_product_attention_flash_musa
    _scaled_dot_product_efficient_attention = None
except ModuleNotFoundError:
    pass

if HAS_AITER:
    from aiter import flash_attn_func as flash_attn_func_aiter

if HAS_FLASH_ATTN:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward

if HAS_FLASHINFER:
    from flashinfer.prefill import single_prefill_with_kv_cache

    _LOG2_E = math.log2(math.e)

if HAS_NPU:
    import torch_npu


def pytorch_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p=0.0,
    softmax_scale=None,
    causal=True,
    window_size=(-1, -1),
    softcap=None,
    alibi_slopes=None,
    return_softmax=False,
    op_type="efficient",
):
    assert op_type in ["flash", "efficient"], f"Invalid op_type: {op_type}"
    """
    q shape (bs, seqlen, nhead, hs)
    k shape (bs, seqlen, nhead, hs)
    v shape (bs, seqlen, nhead, hs)
    """
    # Fallback logic: Flash Attention does not support float32.
    # If op_type is 'flash' but dtype is float32, force 'efficient'.
    if op_type == "flash" and q.dtype == torch.float32:
        op_type = "efficient"

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    if op_type == "flash":
        out, lse = _scaled_dot_product_flash_attention(
            q,
            k,
            v,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale,
        )[:2]
    elif op_type == "efficient":
        out, lse = _scaled_dot_product_efficient_attention(
            q,
            k,
            v,
            attn_bias=None,
            compute_log_sumexp=True,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale,
        )[:2]
    else:
        raise ValueError(f"Invalid op_type: {op_type}")

    out = out.transpose(1, 2)
    lse = lse.to(q.dtype)

    return out, lse


def flash_attn_forward(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=None,
    alibi_slopes=None,
    return_softmax=False,
):
    assert HAS_FLASH_ATTN, "FlashAttention is not available"
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if flash_attn.__version__ < "2.6.3":
        block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax,
        )
    else:
        block_out, block_lse, _, _ = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax,
        )
    return block_out, block_lse


# Cache to track whether flash3_attn_func returns LSE (None = unknown, True/False = tested)
_flash3_returns_lse: bool | None = None


def flash_attn3_func_forward(
    q, k, v, dropout_p, softmax_scale, causal, window_size, softcap, alibi_slopes, return_softmax
):
    """FA3 forward pass for inference (dropout is ignored since FA3 is inference-only)."""
    global _flash3_returns_lse
    assert HAS_FLASH_ATTN_HOPPER, "FA3 (Hopper) is not available"

    # Try high-level flash3_attn_func if available and known to return LSE
    # IMPORTANT: Ring attention's update_out_and_lse requires true LSE values for correct
    # accumulation across ring steps. We must verify the API returns LSE before using it.
    if flash3_attn_func is not None and _flash3_returns_lse is not False:
        # FA3 is inference-only, so we don't pass dropout_p (always 0 for inference)
        result = flash3_attn_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap if softcap else 0.0,
        )

        # Check if result contains valid LSE
        if isinstance(result, tuple) and len(result) > 1 and result[1] is not None:
            if _flash3_returns_lse is None:
                _flash3_returns_lse = True  # Cache: high-level API works
            out, softmax_lse = result[0], result[1]
            return out, softmax_lse
        else:
            # High-level API doesn't return LSE, mark it and fall through to low-level API
            _flash3_returns_lse = False  # Cache: don't try high-level API again

    # Use low-level API which reliably returns LSE
    # Note: fa3_fwd uses different parameter names than flash_attn_interface
    if flash_attn_forward_hopper is not None:
        out, softmax_lse, *unused = flash_attn_forward_hopper(
            q=q,
            k=k,
            v=v,
            k_new=None,
            v_new=None,
            qv=None,
            out_=None,  # fa3_fwd uses out_, not out
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            cu_seqlens_k_new=None,
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=None,
            max_seqlen_k=None,
            page_table=None,
            kv_batch_idx=None,
            leftpad_k=None,
            rotary_cos=None,
            rotary_sin=None,
            seqlens_rotary=None,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0] if window_size else -1,
            window_size_right=window_size[1] if window_size else -1,
            attention_chunk=0,
            softcap=softcap if softcap else 0.0,
            rotary_interleaved=True,
            scheduler_metadata=None,
            num_splits=1,
            pack_gqa=None,
            sm_margin=0,
        )
        return out, softmax_lse

    raise RuntimeError("FA3 is marked as available but no implementation found")


def flash_attn_forward_aiter(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=None,
    alibi_slopes=None,
    return_softmax=False,
):
    assert HAS_AITER, "Aiter is not available"
    block_out, block_lse = flash_attn_func_aiter(
        q,
        k,
        v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        return_lse=True,
    )

    return block_out, block_lse


def flashinfer_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float | None = None,
    alibi_slopes: torch.Tensor | None = None,
    return_softmax: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert HAS_FLASHINFER, "FlashInfer is not available"
    if q.ndim == 4:
        if q.shape[0] > 1:
            raise ValueError("batch size > 1 is not supported")
        out, lse = single_prefill_with_kv_cache(
            q[0],
            k[0],
            v[0],
            sm_scale=softmax_scale,
            causal=causal,
            logits_soft_cap=softcap,
            window_left=window_size[0],
            return_lse=True,
        )
        lse = lse.transpose(0, 1)
        out, lse = out.unsqueeze(0), lse.unsqueeze(0)
    elif q.ndim == 3:
        out, lse = single_prefill_with_kv_cache(
            q,
            k,
            v,
            sm_scale=softmax_scale,
            causal=causal,
            logits_soft_cap=softcap,
            window_left=window_size[0],
            return_lse=True,
        )
        lse = lse.transpose(0, 1)
    else:
        raise ValueError(f"Invalid input shape: {q.shape}")
    lse = lse / _LOG2_E
    return out, lse


def npu_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float | None = None,
    alibi_slopes: torch.Tensor | None = None,
    return_softmax: bool = False,
    layout: str = "BSND",
) -> tuple[torch.Tensor, torch.Tensor]:
    """NPU attention forward compatible with ring attention interface.

    Args:
        q: Query tensor of shape (batch, seq_len, num_heads, head_dim).
        k: Key tensor of shape (batch, seq_len, num_heads, head_dim).
        v: Value tensor of shape (batch, seq_len, num_heads, head_dim).
        dropout_p: Dropout probability (ignored on NPU).
        softmax_scale: Softmax scale factor.
        causal: Causal attention flag (ignored on NPU, handled via pre_tokens/next_tokens).
        window_size: Window size (ignored on NPU).
        softcap: Soft cap value (ignored on NPU).
        alibi_slopes: ALiBi slopes (ignored on NPU).
        return_softmax: Return softmax flag (ignored on NPU).
        layout: Input layout, default "BSND".

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (output, lse).
    """
    assert HAS_NPU, "torch_npu is not available"
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    block_out, block_lse = torch_npu.npu_fused_infer_attention_score(
        q,
        k,
        v,
        num_heads=q.shape[-2],
        input_layout=layout,
        scale=softmax_scale,
        softmax_lse_flag=True,
        pre_tokens=65535,
        next_tokens=65535,
    )
    return block_out, block_lse.squeeze(dim=-1)
