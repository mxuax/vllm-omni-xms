# Parallelism Acceleration Guide

This guide includes how to use parallelism methods in vLLM-Omni to speed up diffusion model inference as well as reduce the memory requirement on each device.

## Overview

The following parallelism methods are currently supported in vLLM-Omni:

1. DeepSpeed Ulysses Sequence Parallel (DeepSpeed Ulysses-SP) ([arxiv paper](https://arxiv.org/pdf/2309.14509)): Ulysses-SP splits the input along the sequence dimension and uses all-to-all communication to allow each device to compute only a subset of attention heads.

2. [Ring-Attention](#ring-attention) - splits the input along the sequence dimension and uses ring-based P2P communication to accumulate attention results, keeping the sequence dimension sharded

3. Classifier-Free-Guidance Parallel (CFG-Parallel): CFG-Parallel runs the positive/negative prompts of classifier-free guidance (CFG) on different devices, then merges on a single device to perform the scheduler step.

4. [Tensor Parallelism](#tensor-parallelism): Tensor parallelism shards model weights across devices. This can reduce per-GPU memory usage. Note that for diffusion models we currently shard the majority of layers within the DiT.

The following table shows which models are currently supported by parallelism method:

### ImageGen

| Model | Model Identifier | Ulysses-SP | Ring-SP | CFG-Parallel | Tensor-Parallel |
|-------|------------------|------------|---------|--------------|--------------------------|
| **LongCat-Image** | `meituan-longcat/LongCat-Image` | ✅ | ✅ | ❌ | ❌ |
| **LongCat-Image-Edit** | `meituan-longcat/LongCat-Image-Edit` | ✅ | ✅ | ❌ | ❌ |
| **Ovis-Image** | `OvisAI/Ovis-Image` | ❌ | ❌ | ❌ | ❌ |
| **Qwen-Image** | `Qwen/Qwen-Image` | ✅ | ✅ | ✅ | ❌ |
| **Qwen-Image-Edit** | `Qwen/Qwen-Image-Edit` | ✅ | ✅ | ✅ | ❌ |
| **Qwen-Image-Edit-2509** | `Qwen/Qwen-Image-Edit-2509` | ✅ | ✅ | ✅ | ❌ |
| **Qwen-Image-Layered** | `Qwen/Qwen-Image-Layered` | ✅ | ✅ | ✅ | ❌ |
| **Z-Image** | `Tongyi-MAI/Z-Image-Turbo` | ❌ | ❌ | ❌ | ✅ (TP=2 only) |
| **Stable-Diffusion3.5** | `stabilityai/stable-diffusion-3.5` | ❌ | ❌ | ❌ | ❌ |

!!! note "Why Z-Image is TP=2 only"
    Z-Image Turbo is currently limited to `tensor_parallel_size` of **1 or 2** due to model shape divisibility constraints.
    For example, the model has `n_heads=30` and a final projection out dimension of `64`, so valid TP sizes must divide both 30 and 64; the only common divisors are **1 and 2**.

### VideoGen

| Model | Model Identifier | Ulysses-SP | Ring-SP | Tensor-Parallel |
|-------|------------------|------------|---------|--------------------------|
| **Wan2.2** | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | ❌ | ❌ | ❌ |

### Tensor Parallelism

Tensor parallelism splits model parameters across GPUs. In vLLM-Omni, tensor parallelism is configured via `DiffusionParallelConfig.tensor_parallel_size`.

#### Offline Inference

```python
from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig

omni = Omni(
    model="Tongyi-MAI/Z-Image-Turbo",
    parallel_config=DiffusionParallelConfig(tensor_parallel_size=2),
)

outputs = omni.generate(
    prompt="a cat reading a book",
    num_inference_steps=9,
    width=512,
    height=512,
)
```

### Sequence Parallelism

#### Ulysses-SP

##### Offline Inference

An example of offline inference script using [Ulysses-SP](https://arxiv.org/pdf/2309.14509) is shown below:
```python
from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig
ulysses_degree = 2

omni = Omni(
    model="Qwen/Qwen-Image",
    parallel_config=DiffusionParallelConfig(ulysses_degree=2)
)

outputs = omni.generate(prompt="A cat sitting on a windowsill", num_inference_steps=50, width=2048, height=2048)
```

See `examples/offline_inference/text_to_image/text_to_image.py` for a complete working example.

##### Online Serving

You can enable Ulysses-SP in online serving for diffusion models via `--usp`:

```bash
# Text-to-image (requires >= 2 GPUs)
vllm serve Qwen/Qwen-Image --omni --port 8091 --usp 2
```

##### Benchmarks
!!! note "Benchmark Disclaimer"
    These benchmarks are provided for **general reference only**. The configurations shown use default or common parameter settings and have not been exhaustively optimized for maximum performance. Actual performance may vary based on:

    - Specific model and use case
    - Hardware configuration
    - Careful parameter tuning
    - Different inference settings (e.g., number of steps, image resolution)


To measure the parallelism methods, we run benchmarks with **Qwen/Qwen-Image** model generating images (**2048x2048** as long sequence input) with 50 inference steps. The hardware devices are NVIDIA H800 GPUs. `sdpa` is the attention backends.

| Configuration | Ulysses degree |Generation Time | Speedup |
|---------------|----------------|---------|---------|
| **Baseline (diffusers)** | - | 112.5s | 1.0x |
| Ulysses-SP  |  2  |  65.2s | 1.73x |
| Ulysses-SP  |  4  | 39.6s | 2.84x |
| Ulysses-SP  |  8  | 30.8s | 3.65x |

#### Ring-Attention

Ring-Attention ([arxiv paper](https://arxiv.org/abs/2310.01889)) splits the input along the sequence dimension and uses ring-based P2P communication to accumulate attention results. Unlike Ulysses-SP which uses all-to-all communication, Ring-Attention keeps the sequence dimension sharded throughout the computation and circulates Key/Value blocks through a ring topology.

##### Offline Inference

An example of offline inference script using Ring-Attention is shown below:
```python
from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig
ring_degree = 2

omni = Omni(
    model="Qwen/Qwen-Image",
    parallel_config=DiffusionParallelConfig(ring_degree=2)
)

outputs = omni.generate(prompt="A cat sitting on a windowsill", num_inference_steps=50, width=2048, height=2048)
```

See `examples/offline_inference/text_to_image/text_to_image.py` for a complete working example.


##### Online Serving

You can enable Ring-Attention in online serving for diffusion models via `--ring`:

```bash
# Text-to-image (requires >= 2 GPUs)
vllm serve Qwen/Qwen-Image --omni --port 8091 --ring 2
```

##### Benchmarks
!!! note "Benchmark Disclaimer"
    These benchmarks are provided for **general reference only**. The configurations shown use default or common parameter settings and have not been exhaustively optimized for maximum performance. Actual performance may vary based on:

    - Specific model and use case
    - Hardware configuration
    - Careful parameter tuning
    - Different inference settings (e.g., number of steps, image resolution)


To measure the parallelism methods, we run benchmarks with **Qwen/Qwen-Image** model generating images (**1024x1024** as long sequence input) with 50 inference steps. The hardware devices are NVIDIA A100 GPUs. `flash_attn` is the attention backends.

| Configuration | Ring degree |Generation Time | Speedup |
|---------------|----------------|---------|---------|
| **Baseline (diffusers)** | - | 45.2s | 1.0x |
| Ring-Attention  |  2  |  29.9s | 1.51x |
| Ring-Attention  |  4  | 23.3s | 1.94x |


#### Hybrid Ulysses + Ring

You can combine both Ulysses-SP and Ring-Attention for larger scale parallelism. The total sequence parallel size equals `ulysses_degree × ring_degree`.

##### Offline Inference

```python
from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig

# Hybrid: 2 Ulysses × 2 Ring = 4 GPUs total
omni = Omni(
    model="Qwen/Qwen-Image",
    parallel_config=DiffusionParallelConfig(ulysses_degree=2, ring_degree=2)
)

outputs = omni.generate(prompt="A cat sitting on a windowsill", num_inference_steps=50, width=2048, height=2048)
```

##### Online Serving

```bash
# Text-to-image (requires >= 4 GPUs)
vllm serve Qwen/Qwen-Image --omni --port 8091 --usp 2 --ring 2
```

##### Benchmarks
!!! note "Benchmark Disclaimer"
    These benchmarks are provided for **general reference only**. The configurations shown use default or common parameter settings and have not been exhaustively optimized for maximum performance. Actual performance may vary based on:

    - Specific model and use case
    - Hardware configuration
    - Careful parameter tuning
    - Different inference settings (e.g., number of steps, image resolution)


To measure the parallelism methods, we run benchmarks with **Qwen/Qwen-Image** model generating images (**1024x1024** as long sequence input) with 50 inference steps. The hardware devices are NVIDIA A100 GPUs. `flash_attn` is the attention backends.

| Configuration | Ulysses degree | Ring degree | Generation Time | Speedup |
|---------------|----------------|-------------|-----------------|---------|
| **Baseline (diffusers)** | - | - | 45.2s | 1.0x |
| Hybrid Ulysses + Ring  |  2  |  2  |  24.3s | 1.87x |


##### How to parallelize a new model

If a diffusion model has been deployed in vLLM-Omni and supports single-card inference, you can enable Context Parallelism (CP) using one of two approaches:

1. **Non-intrusive `_cp_plan` approach** (Recommended): Define a `_cp_plan` class attribute that declaratively specifies how tensors should be sharded/gathered. The framework automatically applies hooks to handle the parallelism.

2. **Intrusive modification approach** (For complex cases): Manually add sharding/gathering logic in the model's `forward()` method.

---

###### Method 1: Non-intrusive `_cp_plan` (Recommended)

The `_cp_plan` mechanism, inspired by the [diffusers library](https://github.com/huggingface/diffusers), allows you to enable CP without modifying the core `forward()` logic. The framework automatically:
- Shards input tensors before computation
- Configures Attention layers for Ulysses/Ring communication
- Gathers output tensors after computation

**Step 1: Understand the `_cp_plan` types**

```python
from vllm_omni.diffusion.distributed.cp_plan import (
    ContextParallelInput,    # For sharding inputs
    ContextParallelOutput,   # For gathering outputs
)

# ContextParallelInput: Shard a tensor along a dimension
ContextParallelInput(
    split_dim=1,           # Dimension to split (usually sequence dim)
    expected_dims=3,       # Expected tensor rank (for validation)
    split_output=False,    # If True, shard the module's OUTPUT instead of input
)

# ContextParallelOutput: Gather sharded outputs
ContextParallelOutput(
    gather_dim=1,          # Dimension to gather
    expected_dims=3,       # Expected tensor rank
)
```

**Step 2: Define `_cp_plan` for your model**

The `_cp_plan` is a dictionary mapping module names to their sharding/gathering specifications:

```python
class MyTransformer2DModel(nn.Module):
    _cp_plan = {
        # Shard inputs to the first transformer block
        "blocks.0": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3),
        },
        # If RoPE module outputs need sharding, use split_output=True
        "rope": {
            0: ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),  # cos
            1: ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),  # sin
        },
        # Gather outputs at the final projection
        "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
    }
```

**Example: Complex model requiring a preparation module (Qwen-Image / Z-Image)**

For models where multiple tensors must be sharded **synchronously** (e.g., `hidden_states` and `vid_freqs` must match in sequence dimension), create a **preparation module** that outputs all tensors together:

```python
# Step 1: Create a preparation module
class ImageRopePrepare(nn.Module):
    """Encapsulates input projection and RoPE computation for _cp_plan sharding."""

    def __init__(self, img_in: nn.Linear, pos_embed: nn.Module):
        super().__init__()
        self.img_in = img_in
        self.pos_embed = pos_embed

    def forward(self, hidden_states, img_shapes, txt_seq_lens):
        hidden_states = self.img_in(hidden_states)
        vid_freqs, txt_freqs = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)
        return hidden_states, vid_freqs, txt_freqs  # All outputs can be sharded together


# Step 2: Define _cp_plan to shard the preparation module's outputs
class QwenImageTransformer2DModel(nn.Module):
    _cp_plan = {
        "image_rope_prepare": {
            0: ContextParallelInput(split_dim=1, expected_dims=3, split_output=True),  # hidden_states
            1: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),  # vid_freqs
            # txt_freqs (index 2) is NOT sharded - kept replicated for dual-stream attention
        },
        "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
    }

    def __init__(self, ...):
        ...
        # Create the preparation module
        self.image_rope_prepare = ImageRopePrepare(self.img_in, self.pos_embed)

    def forward(self, hidden_states, ...):
        # Use the preparation module - _cp_plan will automatically shard its outputs
        hidden_states, vid_freqs, txt_freqs = self.image_rope_prepare(
            hidden_states, img_shapes, txt_seq_lens
        )
        image_rotary_emb = (vid_freqs, txt_freqs)
        ...
```

**Step 3: The framework automatically applies CP**

When the model is loaded with `sequence_parallel_size > 1`, the `registry.py` automatically:
1. Detects the `_cp_plan` attribute
2. Registers hooks to shard/gather tensors according to the plan
3. Configures Attention layers for Ulysses/Ring communication

```python
from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig

omni = Omni(
    model="Qwen/Qwen-Image",
    parallel_config=DiffusionParallelConfig(ulysses_degree=2)  # or ring_degree=2
)

outputs = omni.generate(prompt="A cat sitting on a windowsill", num_inference_steps=50)
```

---

###### Method 2: Intrusive modification (For highly complex cases)

For models with unusual architectures that cannot be expressed via `_cp_plan`, you can manually add sharding logic. This approach requires:

- Manually chunk tensors at the beginning of `forward()`
- Set appropriate forward context flags
- Manually `all_gather` outputs at the end

**Example: Manual sharding in forward()**

```python
from vllm_omni.diffusion.distributed.parallel_state import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
from vllm_omni.diffusion.forward_context import get_forward_context

class ComplexTransformer2DModel(nn.Module):
    def forward(self, hidden_states, encoder_hidden_states, ...):
        sp_size = self.parallel_config.sequence_parallel_size

        if sp_size > 1:
            sp_rank = get_sequence_parallel_rank()
            sp_world_size = get_sequence_parallel_world_size()

            # 1. Shard hidden_states along sequence dimension
            hidden_states = torch.chunk(hidden_states, sp_world_size, dim=1)[sp_rank]

            # 2. Set forward context for attention layers
            # False = text embeddings are replicated (not sharded)
            get_forward_context().split_text_embed_in_sp = False

            # 3. Shard RoPE frequencies to match hidden_states
            img_freqs, txt_freqs = self.pos_embed(...)
            img_freqs = torch.chunk(img_freqs, sp_world_size, dim=0)[sp_rank]
            # txt_freqs kept replicated for dual-stream attention
            image_rotary_emb = (img_freqs, txt_freqs)

        # ... transformer blocks ...

        output = self.proj_out(hidden_states)

        if sp_size > 1:
            # 4. Gather outputs from all ranks
            output = get_sp_group().all_gather(output, dim=1)

        return output
```

**When to use intrusive modification:**

- The model has dynamic sharding logic (e.g., different sharding based on input content)
- Multiple tensors need complex interdependent sharding that can't be expressed as module outputs
- The model requires custom communication patterns beyond simple shard/gather

---

###### Summary: Choosing the right approach

| Scenario | Recommended Approach |
|----------|---------------------|
| Standard transformer with RoPE module | `_cp_plan` with `split_output=True` on rope |
| Dual-stream model (text + image) | Create preparation module + `_cp_plan` |
| Model with complex dynamic sharding | Intrusive modification |
| Rapid prototyping | Intrusive modification (then migrate to `_cp_plan`) |


### CFG-Parallel

##### Offline Inference

CFG-Parallel is enabled through `DiffusionParallelConfig(cfg_parallel_size=...)`. The recommended configuration is `cfg_parallel_size=2` (one rank for the positive branch and one rank for the negative branch).

An example of offline inference using CFG-Parallel (image-to-image) is shown below:

```python
from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig

omni = Omni(
    model="Qwen/Qwen-Image-Edit",
    parallel_config=DiffusionParallelConfig(cfg_parallel_size=2),
)

outputs = omni.generate(
    prompt="turn this cat to a dog",
    negative_prompt="low quality, blurry",
    true_cfg_scale=4.0,
    pil_image=input_image,
    num_inference_steps=50,
)
```

Notes:

- CFG-Parallel is only effective when **true CFG** is enabled (i.e., `true_cfg_scale > 1` and a `negative_prompt` is provided).

#### How to parallelize a pipeline

This section describes how to add CFG-Parallel to a diffusion **pipeline**. We use the Qwen-Image pipeline (`vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image.py`) as the reference implementation.

In `QwenImagePipeline`, each diffusion step runs two denoiser forward passes sequentially:

- positive (prompt-conditioned)
- negative (negative-prompt-conditioned)

CFG-Parallel assigns these two branches to different ranks in the **CFG group** and synchronizes the results.

Below is an example of CFG-Parallel implementation:

```python
def diffuse(
        self,
        ...
        ):
    # Enable CFG-parallel: rank0 computes positive, rank1 computes negative.
    cfg_parallel_ready = do_true_cfg and get_classifier_free_guidance_world_size() > 1

    self.transformer.do_true_cfg = do_true_cfg

    if cfg_parallel_ready:
        cfg_group = get_cfg_group()
        cfg_rank = get_classifier_free_guidance_rank()

        if cfg_rank == 0:
            local_pred = self.transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance,
                encoder_hidden_states_mask=prompt_embeds_mask,
                encoder_hidden_states=prompt_embeds,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                attention_kwargs=self.attention_kwargs,
                return_dict=False,
            )[0]
        else:
            local_pred = self.transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance,
                encoder_hidden_states_mask=negative_prompt_embeds_mask,
                encoder_hidden_states=negative_prompt_embeds,
                img_shapes=img_shapes,
                txt_seq_lens=negative_txt_seq_lens,
                attention_kwargs=self.attention_kwargs,
                return_dict=False,
            )[0]

        gathered = cfg_group.all_gather(local_pred, separate_tensors=True)
        if cfg_rank == 0:
            noise_pred = gathered[0]
            neg_noise_pred = gathered[1]
            comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
            cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
            noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
            noise_pred = comb_pred * (cond_norm / noise_norm)
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        cfg_group.broadcast(latents, src=0)
    else:
        # fallback: run positive then negative sequentially on one rank
        ...
```
