# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Context Parallelism hooks for non-intrusive CP support.

This module implements the hook-based mechanism for applying context parallelism
to models without modifying their forward() methods. It is inspired by diffusers'
_cp_plan approach but uses vLLM-Omni's existing parallel state management.

Usage:
    1. Define _cp_plan on your model class
    2. Call apply_context_parallel(model, config, plan) to enable CP
    3. Call remove_context_parallel(model, plan) to disable CP

The hooks automatically shard inputs before forward and gather outputs after,
based on the plan specification.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.cp_config import ContextParallelConfig
from vllm_omni.diffusion.distributed.cp_plan import (
    ContextParallelInput,
    ContextParallelModelPlan,
    ContextParallelOutput,
)
from vllm_omni.diffusion.distributed.cp_sharding import cp_gather, cp_shard
from vllm_omni.diffusion.hooks.base import HookRegistry, ModelHook

logger = init_logger(__name__)

# Hook name templates for identifying CP hooks
_CP_INPUT_HOOK_TEMPLATE = "cp_input---{}"
_CP_OUTPUT_HOOK_TEMPLATE = "cp_output---{}"


@dataclass
class ModuleForwardMetadata:
    """Metadata for mapping forward() parameter names to args/kwargs positions.

    This caches the inspection of a module's forward signature to efficiently
    locate parameters by name in subsequent calls.
    """

    cached_parameter_indices: dict[str, int] | None = None
    _cls: type | None = None

    def _get_parameter_from_args_kwargs(
        self,
        identifier: str,
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> tuple[Any, bool, int | None]:
        """Get a parameter value from args or kwargs by name.

        Args:
            identifier: The parameter name to look up.
            args: Positional arguments passed to forward.
            kwargs: Keyword arguments passed to forward.

        Returns:
            Tuple of (value, is_kwarg, index).
            - value: The parameter value (or None if not found)
            - is_kwarg: True if found in kwargs
            - index: Position in args if found there

        Raises:
            ValueError: If parameter not found in signature.
        """
        kwargs = kwargs or {}

        # First check kwargs
        if identifier in kwargs:
            return kwargs[identifier], True, None

        # Check cached indices
        if self.cached_parameter_indices is not None:
            index = self.cached_parameter_indices.get(identifier, None)
            if index is None:
                raise ValueError(f"Parameter '{identifier}' not found in cached indices.")
            if index < len(args):
                return args[index], False, index
            return None, False, index

        # Build cache from forward signature
        if self._cls is None:
            raise ValueError("Model class is not set for metadata.")

        parameters = list(inspect.signature(self._cls.forward).parameters.keys())
        parameters = parameters[1:]  # Skip `self`
        self.cached_parameter_indices = {param: i for i, param in enumerate(parameters)}

        if identifier not in self.cached_parameter_indices:
            raise ValueError(f"Parameter '{identifier}' not found in function signature.")

        index = self.cached_parameter_indices[identifier]

        if index >= len(args):
            return None, False, index

        return args[index], False, index


def _unwrap_module(module: nn.Module) -> nn.Module:
    """Unwrap a module from any wrappers to get the original class.

    Args:
        module: Potentially wrapped module.

    Returns:
        The unwrapped module.
    """
    # Handle common wrappers
    while hasattr(module, "_modules") and len(module._modules) == 1:
        inner = next(iter(module._modules.values()))
        if inner is not None:
            module = inner
        else:
            break
    return module


class ContextParallelSplitHook(ModelHook):
    """Hook for splitting inputs before a module's forward pass.

    This hook is registered to modules that need their inputs sharded
    across context parallel ranks. It intercepts the forward call,
    shards specified inputs according to the plan, and passes the
    sharded inputs to the original forward.

    For split_output=True inputs, it shards the output instead.
    """

    def __init__(
        self,
        metadata: dict[str | int, ContextParallelInput | list[ContextParallelInput]],
        config: ContextParallelConfig,
    ) -> None:
        super().__init__()
        self.metadata = metadata
        self.config = config
        self.module_forward_metadata: ModuleForwardMetadata | None = None

    def initialize_hook(self, module: nn.Module) -> nn.Module:
        cls = _unwrap_module(module).__class__
        self.module_forward_metadata = ModuleForwardMetadata(_cls=cls)
        return module

    def pre_forward(self, module: nn.Module, *args: Any, **kwargs: Any) -> tuple[tuple, dict]:
        """Shard inputs before forward."""
        args_list = list(args)

        for name, cpm in self.metadata.items():
            # Skip if this is a split_output entry (handled in post_forward)
            if isinstance(cpm, ContextParallelInput) and cpm.split_output:
                continue

            # Get the parameter value
            input_val, is_kwarg, index = self.module_forward_metadata._get_parameter_from_args_kwargs(
                name, args_list, kwargs
            )

            if input_val is None:
                continue

            # Shard the input
            if isinstance(input_val, torch.Tensor):
                input_val = self._prepare_cp_input(input_val, cpm)
            elif isinstance(input_val, (list, tuple)):
                # Handle list/tuple of tensors with per-element config
                if not isinstance(cpm, (list, tuple)):
                    raise ValueError(
                        f"Expected list/tuple of ContextParallelInput for parameter '{name}' "
                        f"which is a list/tuple, but got {type(cpm).__name__}"
                    )
                if len(input_val) != len(cpm):
                    raise ValueError(f"Expected {len(cpm)} elements for parameter '{name}', got {len(input_val)}")
                sharded_input_val = []
                for i, x in enumerate(input_val):
                    if torch.is_tensor(x) and not cpm[i].split_output:
                        x = self._prepare_cp_input(x, cpm[i])
                    sharded_input_val.append(x)
                input_val = type(input_val)(sharded_input_val)
            else:
                raise ValueError(f"Unsupported input type for sharding: {type(input_val).__name__}")

            # Update args or kwargs
            if is_kwarg:
                kwargs[name] = input_val
            elif index is not None and index < len(args_list):
                args_list[index] = input_val
            else:
                raise ValueError(f"Failed to update parameter '{name}' after sharding.")

        return tuple(args_list), kwargs

    def post_forward(self, module: nn.Module, output: Any) -> Any:
        """Shard outputs for split_output=True entries."""
        is_tensor = isinstance(output, torch.Tensor)
        is_tensor_list = isinstance(output, (list, tuple)) and all(isinstance(x, torch.Tensor) for x in output)

        if not is_tensor and not is_tensor_list:
            # No tensor outputs to shard
            return output

        output_list = [output] if is_tensor else list(output)

        for index, cpm in self.metadata.items():
            if not isinstance(index, int):
                continue
            if not isinstance(cpm, ContextParallelInput) or not cpm.split_output:
                continue
            if index >= len(output_list):
                raise ValueError(f"Index {index} out of bounds for output of length {len(output_list)}.")

            output_list[index] = self._prepare_cp_input(output_list[index], cpm)

        return output_list[0] if is_tensor else type(output)(output_list)

    def _prepare_cp_input(self, x: torch.Tensor, cp_input: ContextParallelInput) -> torch.Tensor:
        """Shard a tensor according to the input specification."""
        if cp_input.expected_dims is not None and x.dim() != cp_input.expected_dims:
            logger.warning_once(f"Expected tensor with {cp_input.expected_dims} dims, got {x.dim()}. Skipping split.")
            return x

        return cp_shard(x, cp_input.split_dim, validate=False)


class ContextParallelGatherHook(ModelHook):
    """Hook for gathering outputs after a module's forward pass.

    This hook is registered to modules that need their outputs gathered
    from all context parallel ranks. It intercepts the output and gathers
    it according to the plan specification.
    """

    def __init__(
        self,
        metadata: ContextParallelOutput | list[ContextParallelOutput],
        config: ContextParallelConfig,
    ) -> None:
        super().__init__()
        if isinstance(metadata, ContextParallelOutput):
            metadata = [metadata]
        self.metadata = metadata
        self.config = config

    def initialize_hook(self, module: nn.Module) -> nn.Module:
        return module

    def post_forward(self, module: nn.Module, output: Any) -> Any:
        """Gather outputs after forward."""
        is_tensor = isinstance(output, torch.Tensor)

        if is_tensor:
            output = [output]
        elif not (isinstance(output, (list, tuple)) and all(isinstance(x, torch.Tensor) for x in output)):
            raise ValueError(f"Expected tensor or list/tuple of tensors, got {type(output).__name__}")

        output = list(output)

        if len(output) != len(self.metadata):
            raise ValueError(f"Expected {len(self.metadata)} outputs, got {len(output)}.")

        for i, cpm in enumerate(self.metadata):
            if cpm is None:
                continue

            x = output[i]
            if cpm.expected_dims is not None and x.dim() != cpm.expected_dims:
                logger.warning_once(
                    f"Expected output tensor with {cpm.expected_dims} dims, got {x.dim()}. Skipping gather."
                )
                continue

            output[i] = cp_gather(x, cpm.gather_dim, validate=False)

        return output[0] if is_tensor else type(output)(output)


def _get_submodule_by_name(model: nn.Module, name: str) -> nn.Module | list[nn.Module]:
    """Get a submodule by dotted name, supporting wildcards.

    Args:
        model: The root module.
        name: Dotted path to submodule. Use "*" to match all children
            of a ModuleList.

    Returns:
        The submodule or list of submodules if wildcard used.

    Raises:
        ValueError: If the path is invalid or module not found.
    """
    if name.count("*") > 1:
        raise ValueError("Wildcard '*' can only be used once in the name")
    return _find_submodule_by_name(model, name)


def _find_submodule_by_name(model: nn.Module, name: str) -> nn.Module | list[nn.Module]:
    """Recursive helper for _get_submodule_by_name."""
    if name == "":
        return model

    first_atom, remaining_name = name.split(".", 1) if "." in name else (name, "")

    if first_atom == "*":
        if not isinstance(model, nn.ModuleList):
            raise ValueError("Wildcard '*' can only be used with ModuleList")
        submodules = []
        for submodule in model:
            subsubmodules = _find_submodule_by_name(submodule, remaining_name)
            if not isinstance(subsubmodules, list):
                subsubmodules = [subsubmodules]
            submodules.extend(subsubmodules)
        return submodules
    else:
        if hasattr(model, first_atom):
            submodule = getattr(model, first_atom)
            return _find_submodule_by_name(submodule, remaining_name)
        else:
            raise ValueError(f"'{first_atom}' is not a submodule of '{model.__class__.__name__}'")


def apply_context_parallel(
    module: nn.Module,
    config: ContextParallelConfig,
    plan: ContextParallelModelPlan,
) -> None:
    """Apply context parallel hooks to a model according to the plan.

    This function registers hooks on the specified submodules to automatically
    shard inputs and gather outputs for context parallelism.

    Args:
        module: The model to apply CP to.
        config: The context parallel configuration.
        plan: Dictionary mapping module names to input/output specifications.

    Example:
        config = ContextParallelConfig(ulysses_degree=2)
        plan = {
            "": {"hidden_states": ContextParallelInput(split_dim=1, expected_dims=3)},
            "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        }
        apply_context_parallel(model, config, plan)
    """
    logger.debug(f"Applying context parallel with config: {config} and plan: {plan}")

    for module_id, cp_model_plan in plan.items():
        submodule = _get_submodule_by_name(module, module_id)
        if not isinstance(submodule, list):
            submodule = [submodule]

        logger.debug(f"Applying CP hooks to '{module_id}' ({len(submodule)} module(s))")

        for m in submodule:
            if isinstance(cp_model_plan, dict):
                # Input specification
                hook = ContextParallelSplitHook(cp_model_plan, config)
                hook_name = _CP_INPUT_HOOK_TEMPLATE.format(module_id)
            elif isinstance(cp_model_plan, (ContextParallelOutput, list, tuple)):
                # Output specification
                if isinstance(cp_model_plan, ContextParallelOutput):
                    cp_model_plan = [cp_model_plan]
                if not all(isinstance(x, ContextParallelOutput) or x is None for x in cp_model_plan):
                    raise ValueError(f"Expected ContextParallelOutput elements, got {cp_model_plan}")
                hook = ContextParallelGatherHook(cp_model_plan, config)
                hook_name = _CP_OUTPUT_HOOK_TEMPLATE.format(module_id)
            else:
                raise ValueError(f"Unsupported plan type: {type(cp_model_plan).__name__}")

            registry = HookRegistry.get_or_create(m)
            registry.register_hook(hook_name, hook)


def remove_context_parallel(
    module: nn.Module,
    plan: ContextParallelModelPlan,
) -> None:
    """Remove context parallel hooks from a model.

    Args:
        module: The model to remove CP from.
        plan: The same plan used when applying CP.
    """
    for module_id, cp_model_plan in plan.items():
        submodule = _get_submodule_by_name(module, module_id)
        if not isinstance(submodule, list):
            submodule = [submodule]

        for m in submodule:
            registry = getattr(m, "_hook_registry", None)
            if registry is None:
                continue

            if isinstance(cp_model_plan, dict):
                hook_name = _CP_INPUT_HOOK_TEMPLATE.format(module_id)
            elif isinstance(cp_model_plan, (ContextParallelOutput, list, tuple)):
                hook_name = _CP_OUTPUT_HOOK_TEMPLATE.format(module_id)
            else:
                continue

            registry.remove_hook(hook_name)


def enable_context_parallel_for_model(
    model: nn.Module,
    config: ContextParallelConfig | None = None,
) -> None:
    """Enable context parallelism for a model using its _cp_plan.

    This is a convenience function that reads the model's _cp_plan attribute
    and applies context parallelism automatically.

    Args:
        model: The model to enable CP for. Must have a _cp_plan attribute.
        config: Optional config. If None, uses default based on current
            parallel state.

    Raises:
        ValueError: If model has no _cp_plan defined.
    """
    from vllm_omni.diffusion.distributed.cp_plan import get_cp_plan_from_model
    from vllm_omni.diffusion.distributed.parallel_state import (
        get_ring_parallel_world_size,
        get_ulysses_parallel_world_size,
    )

    plan = get_cp_plan_from_model(model)
    if plan is None:
        raise ValueError(
            f"Model {model.__class__.__name__} has no _cp_plan defined. "
            f"Define _cp_plan as a class attribute or pass a plan explicitly."
        )

    if config is None:
        # Create config from current parallel state
        config = ContextParallelConfig(
            ulysses_degree=get_ulysses_parallel_world_size(),
            ring_degree=get_ring_parallel_world_size(),
        )

    apply_context_parallel(model, config, plan)


def disable_context_parallel_for_model(model: nn.Module) -> None:
    """Disable context parallelism for a model.

    Args:
        model: The model to disable CP for.
    """
    from vllm_omni.diffusion.distributed.cp_plan import get_cp_plan_from_model

    plan = get_cp_plan_from_model(model)
    if plan is not None:
        remove_context_parallel(model, plan)
