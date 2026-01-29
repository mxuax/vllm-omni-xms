# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)

# Import flash attention functions with fallback chain from utils/fa.py
# FA3 (fa3_fwd_interface) -> FA3 (flash_attn_interface) -> FA2 (flash_attn) -> SDPA fallback
from vllm_omni.diffusion.attention.backends.utils.fa import (
    HAS_FLASH_ATTN,
    _pad_input,
    _unpad_input,
    _upad_input,
    flash_attn_func,
    flash_attn_varlen_func,
)

logger = init_logger(__name__)


class FlashAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @classmethod
    def supports_attention_mask(cls) -> bool:
        return True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 96, 128, 192, 256]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        return FlashAttentionImpl


class FlashAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.num_heads = num_heads
        self.causal = causal
        self.softmax_scale = softmax_scale

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        """CUDA/ROCm flash attention implementation with SDPA fallback."""
        # Use SDPA fallback if no FA backend available
        if not HAS_FLASH_ATTN:
            return self._forward_sdpa(query, key, value, attn_metadata)

        query_length = query.size(1)
        attention_mask = attn_metadata.attn_mask if attn_metadata is not None else None
        #  Contains at least one padding token in the sequence
        if attention_mask is not None and torch.any(~attention_mask):
            assert attention_mask.ndim == 2, "attention_mask must be 2D, (batch_size, seq_len)"
            q, k, v, indices_q, (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = _upad_input(
                query, key, value, attention_mask, query_length, _unpad_input
            )

            out_unpad = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seq_lens_q,
                cu_seqlens_k=cu_seq_lens_k,
                max_seqlen_q=max_length_q,
                max_seqlen_k=max_length_k,
                **{
                    "causal": self.causal,
                    "softmax_scale": self.softmax_scale,
                },
            )
            if isinstance(out_unpad, tuple):
                out_unpad = out_unpad[0]

            out = _pad_input(out_unpad, indices_q, query.size(0), query_length)

        else:
            out = flash_attn_func(
                query,
                key,
                value,
                causal=self.causal,
                softmax_scale=self.softmax_scale,
            )
            # FA3 may return (out, lse) tuple, FA2 returns just out
            if isinstance(out, tuple):
                out = out[0]
        return out

    def _forward_sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        """PyTorch SDPA fallback when no FA backend is available.

        Input shape: (batch, seq_len, num_heads, head_dim)
        SDPA expects: (batch, num_heads, seq_len, head_dim)
        """
        # Transpose to SDPA expected format
        q = query.transpose(1, 2)  # (B, H, S, D)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        # Handle attention mask if present
        attention_mask = attn_metadata.attn_mask if attn_metadata is not None else None
        attn_mask = None
        if attention_mask is not None:
            # Convert boolean mask to attention mask format
            # attention_mask: (B, S) where True = valid, False = masked
            # SDPA expects: (B, 1, 1, S) or (B, 1, S, S) where -inf = masked
            attn_mask = attention_mask[:, None, None, :].float()
            attn_mask = attn_mask.masked_fill(~attention_mask[:, None, None, :], float("-inf"))

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=self.causal if attn_mask is None else False,
            scale=self.softmax_scale,
        )

        # Transpose back to original format
        out = out.transpose(1, 2)  # (B, S, H, D)
        return out

    def forward_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        """NPU attention implementation using mindiesd."""
        from mindiesd import attention_forward

        attention_mask = attn_metadata.attn_mask if attn_metadata else None
        output = attention_forward(
            query,
            key,
            value,
            attn_mask=attention_mask,
            opt_mode="manual",
            op_type="fused_attn_score",
            layout="BNSD",
        )
        return output
