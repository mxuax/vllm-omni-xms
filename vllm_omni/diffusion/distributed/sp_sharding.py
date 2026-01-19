# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project and The HuggingFace Team
"""Sequence Parallelism sharding utilities.

This module provides low-level sharding and gathering functions for
Sequence Parallelism. These can be used directly in model forward methods
for semi-intrusive SP support, or internally by the SP hooks.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.parallel_state import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)

logger = init_logger(__name__)


def sp_shard(
    tensor: torch.Tensor,
    dim: int,
    validate: bool = True,
) -> torch.Tensor:
    """Shard a tensor along the specified dimension for sequence parallelism.

    The tensor is split into world_size chunks along dim, and this rank
    receives its corresponding chunk.

    Args:
        tensor: The tensor to shard.
        dim: The dimension along which to split.
        validate: If True, validate that the tensor size is divisible by world_size.

    Returns:
        The shard for this rank.

    Raises:
        ValueError: If validate=True and tensor size is not divisible by world_size.

    Example:
        # In model forward:
        hidden_states = sp_shard(hidden_states, dim=1)
    """
    world_size = get_sequence_parallel_world_size()

    if world_size == 1:
        return tensor

    rank = get_sequence_parallel_rank()
    size = tensor.size(dim)

    if validate and size % world_size != 0:
        raise ValueError(
            f"Tensor size along dim {dim} ({size}) must be divisible by "
            f"world_size ({world_size}) for sequence parallel sharding."
        )

    return tensor.chunk(world_size, dim=dim)[rank]


def sp_gather(
    tensor: torch.Tensor,
    dim: int,
    validate: bool = True,
) -> torch.Tensor:
    """Gather a tensor along the specified dimension from all sequence parallel ranks.

    The sharded tensors from all ranks are concatenated along dim.

    Args:
        tensor: The local shard to gather.
        dim: The dimension along which to gather.
        validate: If True, validate tensor consistency (currently unused).

    Returns:
        The full tensor gathered from all ranks.

    Example:
        # At end of model forward:
        output = sp_gather(output, dim=1)
    """
    world_size = get_sequence_parallel_world_size()

    if world_size == 1:
        return tensor

    sp_group = get_sp_group()
    return sp_group.all_gather(tensor, dim=dim)


def sp_shard_with_padding(
    tensor: torch.Tensor,
    dim: int,
    pad_value: float = 0.0,
) -> tuple[torch.Tensor, int]:
    """Shard a tensor with automatic padding if not divisible by world_size.

    This is useful for variable-length sequences where padding may be needed.

    Args:
        tensor: The tensor to shard.
        dim: The dimension along which to split.
        pad_value: Value to use for padding.

    Returns:
        Tuple of (sharded_tensor, padding_size). The padding_size indicates
        how much padding was added to the original tensor before sharding.

    Example:
        sharded, pad_size = sp_shard_with_padding(hidden_states, dim=1)
        # ... process ...
        output = sp_gather(output, dim=1)
        if pad_size > 0:
            output = output[..., :-pad_size]  # Remove padding
    """
    world_size = get_sequence_parallel_world_size()

    if world_size == 1:
        return tensor, 0

    size = tensor.size(dim)
    remainder = size % world_size

    if remainder == 0:
        return sp_shard(tensor, dim, validate=False), 0

    # Pad to make divisible
    pad_size = world_size - remainder
    pad_shape = list(tensor.shape)
    pad_shape[dim] = pad_size
    padding = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
    tensor = torch.cat([tensor, padding], dim=dim)

    return sp_shard(tensor, dim, validate=False), pad_size


@dataclass
class ShardingValidator:
    """Validator for tracking and verifying sharding operations.

    This class helps ensure that sharding and gathering operations are
    correctly paired in model forward passes. It tracks which tensors
    have been sharded and verifies that they are properly gathered.

    Usage:
        validator = ShardingValidator()
        with validator.track():
            hidden_states = validator.shard(hidden_states, "hidden_states", dim=1)
            # ... model computation ...
            output = validator.gather(output, "hidden_states", dim=1)
        validator.validate()  # Raises if any shard was not gathered

    Attributes:
        _sharded: Set of tensor names that have been sharded.
        _gathered: Set of tensor names that have been gathered.
        _enabled: Whether tracking is currently enabled.
    """

    _sharded: set[str] = field(default_factory=set)
    _gathered: set[str] = field(default_factory=set)
    _enabled: bool = False

    def reset(self) -> None:
        """Reset the validator state for a new forward pass."""
        self._sharded.clear()
        self._gathered.clear()

    @contextmanager
    def track(self):
        """Context manager to enable tracking for a forward pass."""
        self._enabled = True
        self.reset()
        try:
            yield
        finally:
            self._enabled = False

    def shard(
        self,
        tensor: torch.Tensor,
        name: str,
        dim: int,
        validate_divisible: bool = True,
    ) -> torch.Tensor:
        """Shard a tensor and track the operation.

        Args:
            tensor: The tensor to shard.
            name: A name to identify this tensor for validation.
            dim: The dimension along which to split.
            validate_divisible: If True, validate divisibility.

        Returns:
            The sharded tensor.
        """
        if self._enabled:
            if name in self._sharded:
                logger.warning(f"Tensor '{name}' sharded multiple times")
            self._sharded.add(name)

        return sp_shard(tensor, dim, validate=validate_divisible)

    def gather(
        self,
        tensor: torch.Tensor,
        name: str,
        dim: int,
    ) -> torch.Tensor:
        """Gather a tensor and track the operation.

        Args:
            tensor: The local shard to gather.
            name: The name used when sharding (for validation).
            dim: The dimension along which to gather.

        Returns:
            The gathered tensor.
        """
        if self._enabled:
            if name not in self._sharded:
                logger.warning(f"Tensor '{name}' gathered without being sharded")
            self._gathered.add(name)

        return sp_gather(tensor, dim)

    def validate(self) -> None:
        """Validate that all sharded tensors were gathered.

        Raises:
            ValueError: If any sharded tensor was not gathered.
        """
        unmatched = self._sharded - self._gathered
        if unmatched:
            raise ValueError(
                f"The following tensors were sharded but not gathered: {unmatched}. "
                f"This may indicate a bug in the model's SP implementation."
            )


# Global validator instance for convenience
_global_validator = ShardingValidator()


def get_sharding_validator() -> ShardingValidator:
    """Get the global sharding validator instance.

    Returns:
        The global ShardingValidator.
    """
    return _global_validator


class AllGatherFunction(torch.autograd.Function):
    """Autograd function for all_gather with proper gradient handling.

    This function performs all_gather in the forward pass and reduces
    gradients in the backward pass to maintain gradient correctness
    during training.
    """

    @staticmethod
    def forward(
        ctx: Any,
        tensor: torch.Tensor,
        dim: int,
    ) -> torch.Tensor:
        ctx.dim = dim
        ctx.world_size = get_sequence_parallel_world_size()
        ctx.rank = get_sequence_parallel_rank()
        return sp_gather(tensor, dim, validate=False)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        # Split gradient back to get local portion
        grad_chunks = torch.chunk(grad_output, ctx.world_size, dim=ctx.dim)
        return grad_chunks[ctx.rank].contiguous(), None


def sp_gather_with_grad(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """Gather tensor with proper gradient handling for training.

    Use this instead of sp_gather when gradients need to flow back
    correctly during training.

    Args:
        tensor: The local shard to gather.
        dim: The dimension along which to gather.

    Returns:
        The gathered tensor (autograd-enabled).
    """
    return AllGatherFunction.apply(tensor, dim)
