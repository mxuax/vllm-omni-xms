# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM and The HuggingFace Team
# Type definitions in this module are adapted from HuggingFace diffusers library:
#   diffusers/src/diffusers/models/_modeling_parallel.py
"""Context Parallelism plan type definitions.

This module defines the types used for declaring _cp_plan on model classes,
enabling non-intrusive context parallelism support similar to diffusers.

A _cp_plan is a dictionary that specifies how to shard/gather tensors at
different points in a model's forward pass. This allows automatic handling
of sequence parallelism without modifying the model's forward() method.

Example:
    class MyTransformer(nn.Module):
        _cp_plan = {
            # Split inputs before model forward
            "": {
                "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3),
                "encoder_hidden_states": ContextParallelInput(split_dim=1, expected_dims=3),
            },
            # Split RoPE embeddings after pos_embed layer
            "pos_embed": {
                0: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
            },
            # Gather output after proj_out layer
            "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        }
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextParallelInput:
    """Configuration for splitting an input tensor across context parallel ranks.

    This specifies how to shard a tensor in the pre-forward or post-forward hook
    of a layer. The tensor will be split along the specified dimension.

    Args:
        split_dim: The dimension along which to split the tensor.
        expected_dims: Expected number of dimensions. If provided, validates that
            the tensor has this many dimensions before splitting. If the tensor
            has a different number of dimensions, splitting is skipped with a warning.
        split_output: If True, split the output of the layer instead of the input.
            This is useful for layers whose outputs should be split after preprocessing
            (e.g., RoPE embeddings).

    Example:
        # Split hidden_states along sequence dimension (dim 1)
        ContextParallelInput(split_dim=1, expected_dims=3)

        # Split RoPE output along sequence dimension (dim 0)
        ContextParallelInput(split_dim=0, expected_dims=2, split_output=True)
    """

    split_dim: int
    expected_dims: int | None = None
    split_output: bool = False

    def __repr__(self) -> str:
        return (
            f"ContextParallelInput(split_dim={self.split_dim}, "
            f"expected_dims={self.expected_dims}, split_output={self.split_output})"
        )


@dataclass(frozen=True)
class ContextParallelOutput:
    """Configuration for gathering an output tensor across context parallel ranks.

    This specifies how to gather a tensor in the post-forward hook of a layer.
    The tensor will be gathered along the specified dimension from all ranks.

    Args:
        gather_dim: The dimension along which to gather the tensor.
        expected_dims: Expected number of dimensions. If provided, validates that
            the tensor has this many dimensions before gathering.

    Example:
        # Gather output along sequence dimension (dim 1)
        ContextParallelOutput(gather_dim=1, expected_dims=3)
    """

    gather_dim: int
    expected_dims: int | None = None

    def __repr__(self) -> str:
        return f"ContextParallelOutput(gather_dim={self.gather_dim}, expected_dims={self.expected_dims})"


@dataclass(frozen=True)
class ContextParallelPartialInput:
    """Configuration for partially splitting a tensor (e.g., split image part, keep text part).

    This is designed for models like LongCat/Qwen where RoPE embeddings need special handling:
    - Text portion: kept full across all ranks (for joint attention)
    - Image portion: split across ranks

    The tensor is assumed to be concatenated as [text_part, image_part] along split_dim.

    Args:
        split_dim: The dimension along which to split the image portion.
        text_len_source: How to determine text length:
            - str: Name of a forward parameter that contains text length
            - int: Fixed text length value
        expected_dims: Expected number of dimensions for validation.
        split_output: If True, split the output instead of input.

    Example:
        # Split RoPE: text portion (from txt_ids.shape[0]) kept full, image portion split
        ContextParallelPartialInput(
            split_dim=0,
            text_len_source="txt_ids",  # Get text length from txt_ids.shape[0]
            expected_dims=2,
            split_output=True,
        )

        # Or with fixed text length
        ContextParallelPartialInput(
            split_dim=0,
            text_len_source=512,  # Fixed text length
            expected_dims=2,
            split_output=True,
        )
    """

    split_dim: int
    text_len_source: str | int
    expected_dims: int | None = None
    split_output: bool = False

    def __repr__(self) -> str:
        return (
            f"ContextParallelPartialInput(split_dim={self.split_dim}, "
            f"text_len_source={self.text_len_source!r}, expected_dims={self.expected_dims}, "
            f"split_output={self.split_output})"
        )


# Type aliases for _cp_plan structure

# Any input config type
AnyContextParallelInput = ContextParallelInput | ContextParallelPartialInput

# Input specification: maps parameter names (str) or output indices (int) to split config
ContextParallelInputType = dict[
    str | int,
    AnyContextParallelInput | list[AnyContextParallelInput] | tuple[AnyContextParallelInput, ...],
]

# Output specification: single or multiple gather configs
ContextParallelOutputType = ContextParallelOutput | list[ContextParallelOutput] | tuple[ContextParallelOutput, ...]

# Full model plan: maps module names to input/output specifications
# - Key "" refers to the model itself (root level)
# - Key "module_name" refers to a submodule
# - Key "module_name.*" refers to all children of a ModuleList
ContextParallelModelPlan = dict[str, ContextParallelInputType | ContextParallelOutputType]


def _is_valid_input_config(value: object) -> bool:
    """Check if a value is a valid input configuration type."""
    return isinstance(value, (ContextParallelInput, ContextParallelPartialInput))


def _is_valid_input_config_list(value: object) -> bool:
    """Check if a value is a list/tuple of valid input configurations."""
    if not isinstance(value, (list, tuple)):
        return False
    return all(_is_valid_input_config(x) for x in value)


def validate_cp_plan(plan: ContextParallelModelPlan) -> None:
    """Validate a _cp_plan dictionary for correctness.

    Args:
        plan: The _cp_plan dictionary to validate.

    Raises:
        ValueError: If the plan is invalid.
    """
    if not isinstance(plan, dict):
        raise ValueError(f"_cp_plan must be a dict, got {type(plan).__name__}")

    for module_id, module_plan in plan.items():
        if not isinstance(module_id, str):
            raise ValueError(f"_cp_plan keys must be strings, got {type(module_id).__name__}")

        # Check if it's an output specification (ContextParallelOutput or list/tuple thereof)
        if isinstance(module_plan, ContextParallelOutput):
            continue
        if isinstance(module_plan, (list, tuple)):
            if all(isinstance(x, ContextParallelOutput) for x in module_plan):
                continue
            if _is_valid_input_config_list(module_plan):
                # List of inputs for a specific parameter (when output is tuple)
                continue

        # Otherwise, should be an input specification dict
        if isinstance(module_plan, dict):
            for key, value in module_plan.items():
                if not isinstance(key, (str, int)):
                    raise ValueError(
                        f"Input spec keys must be str or int, got {type(key).__name__} for module '{module_id}'"
                    )
                if isinstance(key, int) and not _is_valid_input_config(value):
                    raise ValueError(
                        f"Integer keys (output indices) must map to ContextParallelInput/PartialInput, "
                        f"got {type(value).__name__} for module '{module_id}'[{key}]"
                    )
                if _is_valid_input_config(value):
                    if isinstance(key, int) and not value.split_output:
                        raise ValueError(
                            f"Integer keys (output indices) require split_output=True, "
                            f"got split_output=False for module '{module_id}'[{key}]"
                        )
                elif _is_valid_input_config_list(value):
                    pass  # Valid list of input configs
                else:
                    raise ValueError(
                        f"Input spec values must be ContextParallelInput/PartialInput or list thereof, "
                        f"got {type(value).__name__} for module '{module_id}'['{key}']"
                    )
        else:
            raise ValueError(
                f"_cp_plan values must be dict (input spec) or ContextParallelOutput, "
                f"got {type(module_plan).__name__} for module '{module_id}'"
            )


def get_cp_plan_from_model(model) -> ContextParallelModelPlan | None:
    """Get the _cp_plan from a model if it exists.

    Args:
        model: The model to get the plan from.

    Returns:
        The _cp_plan dictionary, or None if not defined.
    """
    plan = getattr(model, "_cp_plan", None)
    if plan is not None:
        validate_cp_plan(plan)
    return plan
