# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Simple tests for the CP plan framework.

These tests verify the CP plan mechanism works correctly without requiring
a distributed environment. They test:
1. _cp_plan validation
2. Hook registration and tensor sharding (mocked)
3. Model _cp_plan definitions
"""

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.distributed.cp_plan import (
    ContextParallelInput,
    ContextParallelOutput,
    ContextParallelPartialInput,
    get_cp_plan_from_model,
    validate_cp_plan,
)


class TestContextParallelPlanValidation:
    """Test _cp_plan validation logic."""

    def test_valid_simple_plan(self):
        """Test a simple valid _cp_plan."""
        plan = {
            "rope": {
                0: ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
                1: ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
            },
            "blocks.0": {
                "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3),
            },
            "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        }
        # Should not raise
        validate_cp_plan(plan)

    def test_valid_partial_input_plan(self):
        """Test a valid _cp_plan with ContextParallelPartialInput."""
        plan = {
            "pos_embed": {
                0: ContextParallelPartialInput(
                    split_dim=0,
                    text_len_source="txt_ids",
                    expected_dims=2,
                    split_output=True,
                ),
            },
            "blocks.0": {
                "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3),
            },
        }
        # Should not raise
        validate_cp_plan(plan)

    def test_invalid_plan_type(self):
        """Test that non-dict plan raises error."""
        with pytest.raises(ValueError, match="must be a dict"):
            validate_cp_plan("not a dict")

    def test_invalid_module_key_type(self):
        """Test that non-string module keys raise error."""
        plan = {123: {"hidden_states": ContextParallelInput(split_dim=1)}}
        with pytest.raises(ValueError, match="keys must be strings"):
            validate_cp_plan(plan)

    def test_invalid_output_index_without_split_output(self):
        """Test that integer keys require split_output=True."""
        plan = {
            "rope": {
                0: ContextParallelInput(split_dim=1, split_output=False),  # Invalid
            }
        }
        with pytest.raises(ValueError, match="split_output=True"):
            validate_cp_plan(plan)


class TestGetCpPlanFromModel:
    """Test get_cp_plan_from_model utility."""

    def test_model_with_cp_plan(self):
        """Test getting _cp_plan from a model that has one."""

        class ModelWithPlan(nn.Module):
            _cp_plan = {
                "layer": {
                    "x": ContextParallelInput(split_dim=1),
                }
            }

        model = ModelWithPlan()
        plan = get_cp_plan_from_model(model)
        assert plan is not None
        assert "layer" in plan

    def test_model_without_cp_plan(self):
        """Test getting _cp_plan from a model without one."""

        class ModelWithoutPlan(nn.Module):
            pass

        model = ModelWithoutPlan()
        plan = get_cp_plan_from_model(model)
        assert plan is None


class TestContextParallelInputTypes:
    """Test ContextParallelInput and related types."""

    def test_context_parallel_input_repr(self):
        """Test ContextParallelInput repr."""
        cpi = ContextParallelInput(split_dim=1, expected_dims=3, split_output=True)
        assert "split_dim=1" in repr(cpi)
        assert "expected_dims=3" in repr(cpi)
        assert "split_output=True" in repr(cpi)

    def test_context_parallel_output_repr(self):
        """Test ContextParallelOutput repr."""
        cpo = ContextParallelOutput(gather_dim=1, expected_dims=3)
        assert "gather_dim=1" in repr(cpo)
        assert "expected_dims=3" in repr(cpo)

    def test_context_parallel_partial_input_repr(self):
        """Test ContextParallelPartialInput repr."""
        cppi = ContextParallelPartialInput(
            split_dim=0,
            text_len_source="txt_ids",
            expected_dims=2,
            split_output=True,
        )
        assert "split_dim=0" in repr(cppi)
        assert "txt_ids" in repr(cppi)
        assert "expected_dims=2" in repr(cppi)
        assert "split_output=True" in repr(cppi)

    def test_context_parallel_partial_input_with_int_source(self):
        """Test ContextParallelPartialInput with integer text_len_source."""
        cppi = ContextParallelPartialInput(
            split_dim=0,
            text_len_source=512,  # Fixed length
            expected_dims=2,
        )
        assert cppi.text_len_source == 512


class TestModelCpPlans:
    """Test that model _cp_plan definitions are valid."""

    def test_wan_transformer_cp_plan(self):
        """Test WanTransformer3DModel _cp_plan is valid."""
        try:
            from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanTransformer3DModel

            plan = getattr(WanTransformer3DModel, "_cp_plan", None)
            assert plan is not None, "WanTransformer3DModel should have _cp_plan"
            validate_cp_plan(plan)

            # Check specific entries
            assert "rope" in plan
            assert "blocks.0" in plan
            assert "proj_out" in plan
        except ImportError:
            pytest.skip("WanTransformer3DModel not available")

    def test_zimage_transformer_cp_plan(self):
        """Test ZImageTransformer2DModel _cp_plan is valid."""
        try:
            from vllm_omni.diffusion.models.z_image.z_image_transformer import ZImageTransformer2DModel

            plan = getattr(ZImageTransformer2DModel, "_cp_plan", None)
            assert plan is not None, "ZImageTransformer2DModel should have _cp_plan"
            validate_cp_plan(plan)

            # Check specific entries
            assert "layers.0" in plan
            layers_plan = plan["layers.0"]
            assert "x" in layers_plan
            assert "cos" in layers_plan
            assert "sin" in layers_plan
            assert "attn_mask" in layers_plan
        except ImportError:
            pytest.skip("ZImageTransformer2DModel not available")


class MockShardingTest:
    """Test tensor sharding logic (mocked, no distributed)."""

    def test_shard_tensor_simulation(self):
        """Simulate tensor sharding without distributed backend."""
        # Create a test tensor
        batch_size, seq_len, hidden_dim = 2, 16, 64
        tensor = torch.randn(batch_size, seq_len, hidden_dim)

        # Simulate sharding for world_size=4
        world_size = 4
        rank = 1

        # Manual chunking (what cp_shard does internally)
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
