# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project and The HuggingFace Team
"""Context Parallelism configuration for vLLM-Omni.

This module provides configuration classes for Context Parallelism (CP),
adapting the diffusers-style _cp_plan mechanism to use vLLM-Omni's existing
parallel state management (SequenceParallelGroupCoordinator).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.distributed


@dataclass
class ContextParallelConfig:
    """Configuration for Context Parallelism using vLLM-Omni's parallel state.

    This class provides a unified interface for CP configuration that integrates
    with vLLM-Omni's existing SequenceParallelGroupCoordinator. Unlike diffusers'
    DeviceMesh-based approach, this uses the existing parallel state management.

    Args:
        ulysses_degree: Number of devices for Ulysses (All-to-All) attention.
            Sequence is split across devices, with Q/K/V redistributed via
            All-to-All communication. Best for moderate sequences with good
            interconnect bandwidth.
        ring_degree: Number of devices for Ring attention. Sequence is split
            across devices, with K/V passed in a ring topology. Best for long
            sequences with limited memory/bandwidth.
        convert_to_fp32: Whether to convert output and LSE to float32 for
            numerical stability in ring attention.

    Note:
        ulysses_degree * ring_degree = sequence_parallel_size
        Currently, hybrid Ulysses-Ring attention is supported by vLLM-Omni.
    """

    ulysses_degree: int = 1
    ring_degree: int = 1
    convert_to_fp32: bool = True

    # Internal state - populated by setup()
    _rank: int | None = None
    _world_size: int | None = None
    _device: torch.device | None = None

    def __post_init__(self) -> None:
        if self.ulysses_degree < 1 or self.ring_degree < 1:
            raise ValueError("`ulysses_degree` and `ring_degree` must be >= 1.")

        if self.ulysses_degree == 1 and self.ring_degree == 1:
            raise ValueError(
                "At least one of `ulysses_degree` or `ring_degree` must be > 1 to use context parallelism."
            )

    @property
    def sequence_parallel_size(self) -> int:
        """Total sequence parallel world size."""
        return self.ulysses_degree * self.ring_degree

    def get_world_size(self) -> int:
        """Get the sequence parallel world size from parallel state.

        Returns:
            The world size for sequence parallelism.

        Raises:
            RuntimeError: If parallel state is not initialized.
        """
        from vllm_omni.diffusion.distributed.parallel_state import get_sequence_parallel_world_size

        return get_sequence_parallel_world_size()

    def get_rank(self) -> int:
        """Get the current rank in the sequence parallel group.

        Returns:
            The rank within the sequence parallel group.

        Raises:
            RuntimeError: If parallel state is not initialized.
        """
        from vllm_omni.diffusion.distributed.parallel_state import get_sequence_parallel_rank

        return get_sequence_parallel_rank()

    def get_ulysses_world_size(self) -> int:
        """Get the Ulysses parallel world size.

        Returns:
            The world size for Ulysses (All-to-All) parallelism.
        """
        from vllm_omni.diffusion.distributed.parallel_state import get_ulysses_parallel_world_size

        return get_ulysses_parallel_world_size()

    def get_ulysses_rank(self) -> int:
        """Get the current rank in the Ulysses parallel group.

        Returns:
            The rank within the Ulysses parallel group.
        """
        from vllm_omni.diffusion.distributed.parallel_state import get_ulysses_parallel_rank

        return get_ulysses_parallel_rank()

    def get_ring_world_size(self) -> int:
        """Get the Ring parallel world size.

        Returns:
            The world size for Ring attention parallelism.
        """
        from vllm_omni.diffusion.distributed.parallel_state import get_ring_parallel_world_size

        return get_ring_parallel_world_size()

    def get_ring_rank(self) -> int:
        """Get the current rank in the Ring parallel group.

        Returns:
            The rank within the Ring parallel group.
        """
        from vllm_omni.diffusion.distributed.parallel_state import get_ring_parallel_rank

        return get_ring_parallel_rank()

    def setup(self, rank: int, world_size: int, device: torch.device) -> None:
        """Initialize the config with runtime parallel state.

        This is called automatically when context parallelism is enabled.

        Args:
            rank: The global rank of this process.
            world_size: Total world size.
            device: The device for this rank.
        """
        self._rank = rank
        self._world_size = world_size
        self._device = device

        expected_sp_size = self.ulysses_degree * self.ring_degree
        actual_sp_size = self.get_world_size()

        if expected_sp_size != actual_sp_size:
            raise ValueError(
                f"Configuration mismatch: ulysses_degree ({self.ulysses_degree}) * "
                f"ring_degree ({self.ring_degree}) = {expected_sp_size}, but "
                f"actual sequence parallel world size is {actual_sp_size}."
            )

    def is_initialized(self) -> bool:
        """Check if the config has been initialized with runtime state.

        Returns:
            True if setup() has been called, False otherwise.
        """
        return self._rank is not None
