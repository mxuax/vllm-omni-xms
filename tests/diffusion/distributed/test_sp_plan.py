# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Simple tests for the SP plan framework.

These tests verify the SP plan mechanism works correctly without requiring
a distributed environment. They test:
1. _sp_plan validation
2. Hook registration and tensor sharding (mocked)
3. Model _sp_plan definitions

Note: Our "Sequence Parallelism" (SP) corresponds to "Context Parallelism" (CP) in diffusers.
We use the term "Sequence Parallelism" to align with vLLM-Omni's existing terminology.
"""

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput,
    SequenceParallelOutput,
    SequenceParallelPartialInput,
    get_sp_plan_from_model,
    validate_sp_plan,
)


class TestSequenceParallelPlanValidation:
    """Test _sp_plan validation logic."""

    def test_valid_simple_plan(self):
        """Test a simple valid _sp_plan."""
        plan = {
            "rope": {
                0: SequenceParallelInput(split_dim=1, expected_dims=4, split_output=True),
                1: SequenceParallelInput(split_dim=1, expected_dims=4, split_output=True),
            },
            "blocks.0": {
                "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3),
            },
            "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
        }
        # Should not raise
        validate_sp_plan(plan)

    def test_valid_partial_input_plan(self):
        """Test a valid _sp_plan with SequenceParallelPartialInput."""
        plan = {
            "pos_embed": {
                0: SequenceParallelPartialInput(
                    split_dim=0,
                    text_len_source="txt_ids",
                    expected_dims=2,
                    split_output=True,
                ),
            },
            "blocks.0": {
                "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3),
            },
        }
        # Should not raise
        validate_sp_plan(plan)

    def test_invalid_plan_type(self):
        """Test that non-dict plan raises error."""
        with pytest.raises(ValueError, match="must be a dict"):
            validate_sp_plan("not a dict")

    def test_invalid_module_key_type(self):
        """Test that non-string module keys raise error."""
        plan = {123: {"hidden_states": SequenceParallelInput(split_dim=1)}}
        with pytest.raises(ValueError, match="keys must be strings"):
            validate_sp_plan(plan)

    def test_invalid_output_index_without_split_output(self):
        """Test that integer keys require split_output=True."""
        plan = {
            "rope": {
                0: SequenceParallelInput(split_dim=1, split_output=False),  # Invalid
            }
        }
        with pytest.raises(ValueError, match="split_output=True"):
            validate_sp_plan(plan)


class TestGetSpPlanFromModel:
    """Test get_sp_plan_from_model utility."""

    def test_model_with_sp_plan(self):
        """Test getting _sp_plan from a model that has one."""

        class ModelWithPlan(nn.Module):
            _sp_plan = {
                "layer": {
                    "x": SequenceParallelInput(split_dim=1),
                }
            }

        model = ModelWithPlan()
        plan = get_sp_plan_from_model(model)
        assert plan is not None
        assert "layer" in plan

    def test_model_without_sp_plan(self):
        """Test getting _sp_plan from a model without one."""

        class ModelWithoutPlan(nn.Module):
            pass

        model = ModelWithoutPlan()
        plan = get_sp_plan_from_model(model)
        assert plan is None


class TestSequenceParallelInputTypes:
    """Test SequenceParallelInput and related types."""

    def test_sequence_parallel_input_repr(self):
        """Test SequenceParallelInput repr."""
        spi = SequenceParallelInput(split_dim=1, expected_dims=3, split_output=True)
        assert "split_dim=1" in repr(spi)
        assert "expected_dims=3" in repr(spi)
        assert "split_output=True" in repr(spi)

    def test_sequence_parallel_output_repr(self):
        """Test SequenceParallelOutput repr."""
        spo = SequenceParallelOutput(gather_dim=1, expected_dims=3)
        assert "gather_dim=1" in repr(spo)
        assert "expected_dims=3" in repr(spo)

    def test_sequence_parallel_partial_input_repr(self):
        """Test SequenceParallelPartialInput repr."""
        sppi = SequenceParallelPartialInput(
            split_dim=0,
            text_len_source="txt_ids",
            expected_dims=2,
            split_output=True,
        )
        assert "split_dim=0" in repr(sppi)
        assert "txt_ids" in repr(sppi)
        assert "expected_dims=2" in repr(sppi)
        assert "split_output=True" in repr(sppi)

    def test_sequence_parallel_partial_input_with_int_source(self):
        """Test SequenceParallelPartialInput with integer text_len_source."""
        sppi = SequenceParallelPartialInput(
            split_dim=0,
            text_len_source=512,  # Fixed length
            expected_dims=2,
        )
        assert sppi.text_len_source == 512


class TestModelSpPlans:
    """Test that model _sp_plan definitions are valid."""

    def test_zimage_transformer_sp_plan(self):
        """Test ZImageTransformer2DModel _sp_plan structure.

        The plan specifies:
        - unified_prepare: Shard all 4 outputs (unified, cos, sin, attn_mask)
        - all_final_layer.2-1: Gather outputs after final layer

        Note: _sp_plan corresponds to diffusers' _cp_plan (Context Parallelism)
        """
        try:
            from vllm_omni.diffusion.models.z_image.z_image_transformer import ZImageTransformer2DModel

            plan = getattr(ZImageTransformer2DModel, "_sp_plan", None)
            assert plan is not None, "ZImageTransformer2DModel should define _sp_plan"
            assert isinstance(plan, dict)

            assert "unified_prepare" in plan
            unified_prepare_plan = plan["unified_prepare"]
            # Check all 4 outputs are sharded with split_output=True
            assert 0 in unified_prepare_plan  # unified
            assert 1 in unified_prepare_plan  # unified_cos
            assert 2 in unified_prepare_plan  # unified_sin
            assert 3 in unified_prepare_plan  # unified_attn_mask

            # Check output gathering
            assert "all_final_layer.2-1" in plan
        except ImportError:
            pytest.skip("ZImageTransformer2DModel not available")

    def test_qwen_image_transformer_sp_plan(self):
        """Test QwenImageTransformer2DModel _sp_plan structure.

        Qwen-Image follows the diffusers pattern similar to Z-Image:
        - image_rope_prepare: Shards hidden_states and vid_freqs together
        - proj_out: Gathers output

        Key insight: hidden_states and vid_freqs MUST be sharded together
        to maintain dimension alignment for RoPE computation.

        Note: _sp_plan corresponds to diffusers' _cp_plan (Context Parallelism)
        """
        try:
            from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
                QwenImageTransformer2DModel,
            )

            plan = getattr(QwenImageTransformer2DModel, "_sp_plan", None)
            assert plan is not None, "QwenImageTransformer2DModel should define _sp_plan"
            assert isinstance(plan, dict)

            # Check image_rope_prepare sharding
            assert "image_rope_prepare" in plan
            rope_plan = plan["image_rope_prepare"]
            # hidden_states (index 0)
            assert 0 in rope_plan
            assert rope_plan[0].split_dim == 1
            assert rope_plan[0].split_output is True
            # vid_freqs (index 1)
            assert 1 in rope_plan
            assert rope_plan[1].split_dim == 0
            assert rope_plan[1].split_output is True
            # txt_freqs (index 2) should NOT be in plan (kept replicated)
            assert 2 not in rope_plan

            # Check output gathering at proj_out
            assert "proj_out" in plan
            proj_out_plan = plan["proj_out"]
            assert proj_out_plan.gather_dim == 1

            validate_sp_plan(plan)
        except ImportError:
            pytest.skip("QwenImageTransformer2DModel not available")


class TestMockSharding:
    """Test tensor sharding logic (mocked, no distributed)."""

    def test_shard_tensor_simulation(self):
        """Simulate tensor sharding without distributed backend."""
        # Create a test tensor
        batch_size, seq_len, hidden_dim = 2, 16, 64
        tensor = torch.randn(batch_size, seq_len, hidden_dim)

        # Simulate sharding for world_size=4
        world_size = 4
        rank = 1

        # Manual chunking (what sp_shard does internally)
        chunks = tensor.chunk(world_size, dim=1)
        sharded = chunks[rank]

        assert sharded.shape == (batch_size, seq_len // world_size, hidden_dim)
        assert sharded.shape == (2, 4, 64)

    def test_partial_shard_simulation(self):
        """Simulate partial sharding (text kept, image sharded)."""
        # Create a test tensor with [text, image] concatenated
        batch_size = 2
        text_len = 8
        image_len = 16
        hidden_dim = 64

        text_part = torch.randn(batch_size, text_len, hidden_dim)
        image_part = torch.randn(batch_size, image_len, hidden_dim)
        tensor = torch.cat([text_part, image_part], dim=1)

        assert tensor.shape == (batch_size, text_len + image_len, hidden_dim)

        # Simulate partial sharding for world_size=4, rank=1
        world_size = 4
        rank = 1
        dim = 1

        # Extract parts
        text_kept = tensor.narrow(dim, 0, text_len)
        image_full = tensor.narrow(dim, text_len, image_len)

        # Shard only image part
        image_chunks = image_full.chunk(world_size, dim=dim)
        image_sharded = image_chunks[rank]

        # Concatenate back
        result = torch.cat([text_kept, image_sharded], dim=dim)

        expected_len = text_len + image_len // world_size
        assert result.shape == (batch_size, expected_len, hidden_dim)
        assert result.shape == (2, 8 + 4, 64)  # text_len + image_len/4
