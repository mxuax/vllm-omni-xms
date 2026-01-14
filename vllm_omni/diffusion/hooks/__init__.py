# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hook mechanism for model forward interception."""

from vllm_omni.diffusion.hooks.base import (
    BaseState,
    HookRegistry,
    ModelHook,
    StateManager,
)
from vllm_omni.diffusion.hooks.context_parallel import (
    ContextParallelGatherHook,
    ContextParallelSplitHook,
    apply_context_parallel,
    disable_context_parallel_for_model,
    enable_context_parallel_for_model,
    remove_context_parallel,
)

__all__ = [
    # Base hooks
    "BaseState",
    "StateManager",
    "ModelHook",
    "HookRegistry",
    # Context parallel hooks
    "ContextParallelSplitHook",
    "ContextParallelGatherHook",
    "apply_context_parallel",
    "remove_context_parallel",
    "enable_context_parallel_for_model",
    "disable_context_parallel_for_model",
]
