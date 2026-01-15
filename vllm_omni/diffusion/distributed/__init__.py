# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Distributed utilities for vLLM-Omni diffusion models."""

from vllm_omni.diffusion.distributed.cp_config import ContextParallelConfig
from vllm_omni.diffusion.distributed.cp_plan import (
    ContextParallelInput,
    ContextParallelModelPlan,
    ContextParallelOutput,
    ContextParallelPartialInput,
    get_cp_plan_from_model,
    validate_cp_plan,
)
from vllm_omni.diffusion.distributed.cp_sharding import (
    AllGatherFunction,
    ShardingValidator,
    cp_gather,
    cp_gather_with_grad,
    cp_shard,
    cp_shard_with_padding,
    get_sharding_validator,
)

__all__ = [
    # Config
    "ContextParallelConfig",
    # Plan types
    "ContextParallelInput",
    "ContextParallelOutput",
    "ContextParallelPartialInput",
    "ContextParallelModelPlan",
    "validate_cp_plan",
    "get_cp_plan_from_model",
    # Sharding utilities
    "cp_shard",
    "cp_gather",
    "cp_shard_with_padding",
    "cp_gather_with_grad",
    "AllGatherFunction",
    "ShardingValidator",
    "get_sharding_validator",
]
