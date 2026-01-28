# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) Microsoft Corporation and Jiarui Fang
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team & Jiarui Fang
# Adapted from
# https://github.com/feifeibear/long-context-attention/blob/main/yunchang/attention/layer.py


import torch
import torch.nn as nn
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.parallel import build_parallel_attention_strategy
from vllm_omni.diffusion.attention.parallel.ring import RingParallelAttention
from vllm_omni.diffusion.attention.selector import get_attn_backend
from vllm_omni.diffusion.forward_context import get_forward_context

logger = init_logger(__name__)


class Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        # ulysses attention
        scatter_idx: int = 2,
        gather_idx: int = 1,
        use_sync: bool = False,
    ):
        super().__init__()
        self.attn_backend = get_attn_backend(-1)
        self.attn_impl_cls = self.attn_backend.get_impl_cls()
        self.attention = self.attn_impl_cls(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
        )
        self.backend_pref = None

        self.softmax_scale = softmax_scale
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.use_sync = use_sync
        self.causal = causal

        try:
            config = get_forward_context().omni_diffusion_config
            self.backend_pref = config.attention_backend
        except Exception:
            pass

        # Build parallel strategy (handles Ulysses, Ring, or NoParallel)
        self.parallel_strategy = build_parallel_attention_strategy(
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            use_sync=use_sync,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        # 1. Prepare inputs (Communication / Resharding)
        # For Ulysses: AllToAll Q/K/V; Slicing joint_q/k/v
        # For Ring: Concat joint_q
        query, key, value, attn_metadata, ctx = self.parallel_strategy.pre_attention(query, key, value, attn_metadata)

        # 2. Kernel Execution (Computation)
        # Ring attention uses its own kernel with P2P communication
        # Ulysses and NoParallel use local attention kernel
        if isinstance(self.parallel_strategy, RingParallelAttention):
            out = self.parallel_strategy.run_attention(
                query, key, value, attn_metadata, softmax_scale=self.softmax_scale, causal=self.causal
            )
        else:
            out = self._run_local_attention(query, key, value, attn_metadata)

        # 3. Post-processing (Reverse Communication)
        # For Ulysses: AllToAll Output, and AllGather Joint Output
        # For Ring/NoParallel: no-op
        out = self.parallel_strategy.post_attention(out, ctx)

        return out

    def _run_local_attention(self, query, key, value, attn_metadata):
        if self.backend_pref == "flash_attn" and query.dtype == torch.float32:
            logger.warning(
                "Flash Attention does not support float32. Overriding user config "
                f"attention_backend='{self.backend_pref}' to 'sdpa' for dtype={query.dtype}."
            )
            self.backend_pref = "sdpa"

        # Fallback to standard attention
        return self.attention.forward(query, key, value, attn_metadata)
