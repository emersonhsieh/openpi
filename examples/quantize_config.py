"""Quantization utilities for PI0 model conversion.

Configurable MXU/VPU quantization scheme for JAX-to-PyTorch conversion.

MXU (Matrix Multiply Unit): weight matrices in linear/conv layers.
VPU (Vector Processing Unit): biases, normalization parameters, embeddings.

Constraint: MXU output dtype must equal VPU input dtype, since data flows
from matmul outputs directly into vector operations (e.g., LayerNorm).

Supported dtypes: E4M3, FP16, BF16, FP32.

Usage:
    # Default (MXU input=E4M3, MXU output=BF16, VPU input=BF16, VPU output=BF16):
    python examples/quantize_config.py --checkpoint_dir /path/to/ckpt --config_name pi0_droid --output_path /path/to/out

    # Custom quantization:
    python examples/quantize_config.py --checkpoint_dir /path/to/ckpt --config_name pi0_droid --output_path /path/to/out \\
        --mxu-input e4m3 --mxu-output bf16 --vpu-input bf16 --vpu-output bf16

    # Inspect only:
    python examples/quantize_config.py --checkpoint_dir /path/to/ckpt --config_name pi0_droid --inspect_only
"""

import json
import os
import pathlib
import shutil
from enum import Enum
from typing import Literal

from flax.nnx import traversals
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch
import tyro

import openpi.models.gemma
import openpi.models.model
import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
from openpi.training import utils
import openpi.training.config as _config

# ---------------------------------------------------------------------------
# JAX-to-PyTorch conversion helpers (copied from convert_jax_model_to_pytorch.py)
# ---------------------------------------------------------------------------


def slice_paligemma_state_dict(state_dict, config):
    """Convert PaliGemma JAX parameters to PyTorch format."""
    suffix = "/value" if "img/embedding/kernel/value" in state_dict else ""

    # patch embeddings
    jax_key = f"img/embedding/kernel{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose(3, 2, 0, 1)

    jax_key = f"img/embedding/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.bias"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # positional embeddings
    jax_key = f"img/pos_embedding{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.position_embedding.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).reshape(-1, config.vision_config.hidden_size)

    # extract vision layers to be sliced at index 0. There are 27 layers in the base model.
    encoderblock_layernorm0_scale = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_0/scale{suffix}")
    encoderblock_layernorm0_bias = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_0/bias{suffix}")
    encoderblock_layernorm1_scale = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_1/scale{suffix}")
    encoderblock_layernorm1_bias = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_1/bias{suffix}")

    encoderblock_mlp_dense0_kernel = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel{suffix}")
    encoderblock_mlp_dense0_bias = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias{suffix}")
    encoderblock_mlp_dense1_kernel = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel{suffix}")
    encoderblock_mlp_dense1_bias = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias{suffix}")

    encoderblock_attention_0_key_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel{suffix}"
    )
    encoderblock_attention_0_key_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias{suffix}"
    )
    encoderblock_attention_0_value_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel{suffix}"
    )
    encoderblock_attention_0_value_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias{suffix}"
    )
    encoderblock_attention_0_query_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel{suffix}"
    )
    encoderblock_attention_0_query_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias{suffix}"
    )
    encoderblock_attention_0_out_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel{suffix}"
    )
    encoderblock_attention_0_out_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias{suffix}"
    )

    for i in range(config.vision_config.num_hidden_layers):
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.weight"
        ] = encoderblock_layernorm0_scale[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.bias"
        ] = encoderblock_layernorm0_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.weight"
        ] = encoderblock_layernorm1_scale[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.bias"
        ] = encoderblock_layernorm1_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.weight"
        ] = encoderblock_mlp_dense0_kernel[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.bias"
        ] = encoderblock_mlp_dense0_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.weight"
        ] = encoderblock_mlp_dense1_kernel[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.bias"
        ] = encoderblock_mlp_dense1_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.weight"
        ] = encoderblock_attention_0_key_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.bias"
        ] = encoderblock_attention_0_key_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.weight"
        ] = encoderblock_attention_0_value_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.bias"
        ] = encoderblock_attention_0_value_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.weight"
        ] = encoderblock_attention_0_query_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.bias"
        ] = encoderblock_attention_0_query_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.weight"
        ] = encoderblock_attention_0_out_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.bias"
        ] = encoderblock_attention_0_out_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)

    jax_key = f"img/Transformer/encoder_norm/scale{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose()

    jax_key = f"img/Transformer/encoder_norm/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.bias"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # multimodal projector
    jax_key = f"img/head/kernel{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose()

    jax_key = f"img/head/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # text decoder (gemma)
    jax_key = f"llm/embedder/input_embedding{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # pop the einsum attention + mlp representations
    llm_attention_attn_vec_einsum = state_dict.pop(f"llm/layers/attn/attn_vec_einsum/w{suffix}")
    llm_attention_kv_einsum = state_dict.pop(f"llm/layers/attn/kv_einsum/w{suffix}")
    llm_attention_q_einsum = state_dict.pop(f"llm/layers/attn/q_einsum/w{suffix}")

    llm_mlp_gating_einsum = state_dict.pop(f"llm/layers/mlp/gating_einsum{suffix}")
    llm_mlp_linear = state_dict.pop(f"llm/layers/mlp/linear{suffix}")

    llm_input_layernorm = state_dict.pop(f"llm/layers/pre_attention_norm/scale{suffix}")
    llm_post_attention_layernorm = state_dict.pop(f"llm/layers/pre_ffw_norm/scale{suffix}")

    for i in range(config.text_config.num_hidden_layers):
        q_proj_weight_reshaped = (
            llm_attention_q_einsum[i]
            .transpose(0, 2, 1)
            .reshape(
                config.text_config.num_attention_heads * config.text_config.head_dim, config.text_config.hidden_size
            )
        )
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.q_proj.weight"] = (
            q_proj_weight_reshaped
        )

        k_proj_weight_reshaped = llm_attention_kv_einsum[i, 0, 0].transpose()
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.k_proj.weight"] = (
            k_proj_weight_reshaped
        )
        v_proj_weight_reshaped = llm_attention_kv_einsum[i, 1, 0].transpose()
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.v_proj.weight"] = (
            v_proj_weight_reshaped
        )

        o_proj_weight_reshaped = (
            llm_attention_attn_vec_einsum[i]
            .transpose(2, 0, 1)
            .reshape(
                config.text_config.num_attention_heads * config.text_config.head_dim, config.text_config.hidden_size
            )
        )
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.o_proj.weight"] = (
            o_proj_weight_reshaped
        )

        gate_proj_weight = llm_mlp_gating_einsum[i, 0]
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.gate_proj.weight"] = (
            gate_proj_weight.transpose()
        )
        up_proj_weight = llm_mlp_gating_einsum[i, 1]
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.up_proj.weight"] = (
            up_proj_weight.transpose()
        )
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.down_proj.weight"] = (
            llm_mlp_linear[i].transpose()
        )
        state_dict[f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.input_layernorm.weight"] = (
            llm_input_layernorm[i]
        )
        state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.post_attention_layernorm.weight"
        ] = llm_post_attention_layernorm[i]

    jax_key = f"llm/final_norm/scale{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.language_model.norm.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    expert_dict = {}
    final_state_dict = {}

    # Expert-related keys to extract (including pi05 Dense layer parameters)
    expert_keys = [
        f"llm/final_norm_1/scale{suffix}",
        f"llm/final_norm_1/Dense_0/bias{suffix}",
        f"llm/final_norm_1/Dense_0/kernel{suffix}",
        f"llm/layers/attn/attn_vec_einsum_1/w{suffix}",
        f"llm/layers/attn/kv_einsum_1/w{suffix}",
        f"llm/layers/attn/q_einsum_1/w{suffix}",
        f"llm/layers/mlp_1/gating_einsum{suffix}",
        f"llm/layers/mlp_1/linear{suffix}",
        f"llm/layers/pre_attention_norm_1/scale{suffix}",
        f"llm/layers/pre_attention_norm_1/Dense_0/bias{suffix}",
        f"llm/layers/pre_attention_norm_1/Dense_0/kernel{suffix}",
        f"llm/layers/pre_ffw_norm_1/scale{suffix}",
        f"llm/layers/pre_ffw_norm_1/Dense_0/bias{suffix}",
        f"llm/layers/pre_ffw_norm_1/Dense_0/kernel{suffix}",
    ]

    for key, value in state_dict.items():
        if key not in expert_keys:
            final_state_dict[key] = torch.from_numpy(value)
        else:
            expert_dict[key] = value

    return final_state_dict, expert_dict


def slice_gemma_state_dict(state_dict, config, *, num_expert, checkpoint_dir, pi05):
    """Convert Gemma JAX parameters to PyTorch format."""
    # Add missing attributes to config if they don't exist
    if not hasattr(config, "vocab_size"):
        config.vocab_size = 257152  # PALIGEMMA_VOCAB_SIZE
    if not hasattr(config, "hidden_size"):
        config.hidden_size = config.width
    if not hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = config.depth
    if not hasattr(config, "num_attention_heads"):
        config.num_attention_heads = config.num_heads

    suffix = "/value" if f"llm/layers/attn/attn_vec_einsum_{num_expert}/w/value" in state_dict else ""

    llm_attention_attn_vec_einsum = state_dict.pop(f"llm/layers/attn/attn_vec_einsum_{num_expert}/w{suffix}")
    llm_attention_kv_einsum = state_dict.pop(f"llm/layers/attn/kv_einsum_{num_expert}/w{suffix}")
    llm_attention_q_einsum = state_dict.pop(f"llm/layers/attn/q_einsum_{num_expert}/w{suffix}")

    llm_mlp_gating_einsum = state_dict.pop(f"llm/layers/mlp_{num_expert}/gating_einsum{suffix}")
    llm_mlp_linear = state_dict.pop(f"llm/layers/mlp_{num_expert}/linear{suffix}")

    # Check if we have Dense layers (for pi05/adaptive normalization) or scale layers (for regular pi0)
    if "pi05" in checkpoint_dir:
        # Pi05 with adaptive normalization
        llm_input_layernorm_bias = state_dict.pop(f"llm/layers/pre_attention_norm_{num_expert}/Dense_0/bias{suffix}")
        llm_post_attention_layernorm_bias = state_dict.pop(f"llm/layers/pre_ffw_norm_{num_expert}/Dense_0/bias{suffix}")
        llm_input_layernorm_kernel = state_dict.pop(
            f"llm/layers/pre_attention_norm_{num_expert}/Dense_0/kernel{suffix}"
        )
        llm_post_attention_layernorm_kernel = state_dict.pop(
            f"llm/layers/pre_ffw_norm_{num_expert}/Dense_0/kernel{suffix}"
        )
    else:
        # Regular pi0 with standard RMSNorm
        llm_input_layernorm = state_dict.pop(f"llm/layers/pre_attention_norm_{num_expert}/scale{suffix}")
        llm_post_attention_layernorm = state_dict.pop(f"llm/layers/pre_ffw_norm_{num_expert}/scale{suffix}")

    for i in range(config.num_hidden_layers):
        q_proj_weight_reshaped = (
            llm_attention_q_einsum[i]
            .transpose(0, 2, 1)
            .reshape(config.num_attention_heads * config.head_dim, config.hidden_size)
        )
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.q_proj.weight"] = (
            q_proj_weight_reshaped
        )

        k_proj_weight_reshaped = llm_attention_kv_einsum[i, 0, 0].transpose()
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.k_proj.weight"] = (
            k_proj_weight_reshaped
        )
        v_proj_weight_reshaped = llm_attention_kv_einsum[i, 1, 0].transpose()
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.v_proj.weight"] = (
            v_proj_weight_reshaped
        )

        o_proj_weight_reshaped = (
            llm_attention_attn_vec_einsum[i]
            .reshape(config.num_attention_heads * config.head_dim, config.hidden_size)
            .transpose(1, 0)
        )
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.o_proj.weight"] = (
            o_proj_weight_reshaped
        )

        gate_proj_weight = llm_mlp_gating_einsum[i, 0]
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.gate_proj.weight"] = (
            gate_proj_weight.transpose()
        )
        up_proj_weight = llm_mlp_gating_einsum[i, 1]
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.up_proj.weight"] = (
            up_proj_weight.transpose()
        )
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.down_proj.weight"] = llm_mlp_linear[
            i
        ].transpose()

        if "pi05" in checkpoint_dir:
            # Pi05 with adaptive normalization - use Dense layer parameters directly
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.dense.bias"] = (
                llm_input_layernorm_bias[i]
            )
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.dense.bias"] = (
                llm_post_attention_layernorm_bias[i]
            )
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.dense.weight"] = (
                llm_input_layernorm_kernel[i].transpose()
            )
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.dense.weight"] = (
                llm_post_attention_layernorm_kernel[i].transpose()
            )
        else:
            # Regular pi0 with standard RMSNorm
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.weight"] = (
                llm_input_layernorm[i]
            )
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.weight"] = (
                llm_post_attention_layernorm[i]
            )

    # Handle final norm layer
    if "pi05" in checkpoint_dir:
        # Pi05 with adaptive normalization - use Dense layer parameters directly
        final_norm_bias = state_dict.pop(f"llm/final_norm_{num_expert}/Dense_0/bias{suffix}")
        final_norm_kernel = state_dict.pop(f"llm/final_norm_{num_expert}/Dense_0/kernel{suffix}")
        state_dict["paligemma_with_expert.gemma_expert.model.norm.dense.bias"] = final_norm_bias
        state_dict["paligemma_with_expert.gemma_expert.model.norm.dense.weight"] = final_norm_kernel.transpose()
    else:
        # Regular pi0 with standard RMSNorm
        state_dict["paligemma_with_expert.gemma_expert.model.norm.weight"] = state_dict.pop(
            f"llm/final_norm_{num_expert}/scale{suffix}"
        )

    final_state_dict = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            final_state_dict[key] = torch.from_numpy(value)
        else:
            final_state_dict[key] = value

    return final_state_dict


def slice_initial_orbax_checkpoint(checkpoint_dir: str, restore_precision: str | None = None):
    """Load and process params by restoring via JAX model loader first.
    This respects dtype conversions that occur during model restore.
    """
    params = openpi.models.model.restore_params(
        f"{checkpoint_dir}/params/", restore_type=np.ndarray, dtype=restore_precision
    )

    return {"paligemma_params": traversals.flatten_mapping(params["PaliGemma"], sep="/"), "projection_params": params}


def load_jax_model_and_print_keys(checkpoint_dir: str):
    """Load JAX model from checkpoint and print all parameter keys."""
    checkpoint_dir = os.path.abspath(checkpoint_dir) if not checkpoint_dir.startswith("gs://") else checkpoint_dir
    checkpointer = ocp.PyTreeCheckpointer()
    metadata = checkpointer.metadata(f"{checkpoint_dir}/params")
    print(utils.array_tree_to_info(metadata))


# ---------------------------------------------------------------------------
# Quantization dtype enum & mapping
# ---------------------------------------------------------------------------

class QuantDtype(str, Enum):
    """Supported quantization dtypes."""
    E4M3 = "e4m3"
    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"


QDTYPE_TO_TORCH: dict[QuantDtype, torch.dtype] = {
    QuantDtype.E4M3: torch.float8_e4m3fn,
    QuantDtype.FP16: torch.float16,
    QuantDtype.BF16: torch.bfloat16,
    QuantDtype.FP32: torch.float32,
}


# ---------------------------------------------------------------------------
# Parameter classification
# ---------------------------------------------------------------------------

def is_mxu_weight(key: str) -> bool:
    """Classify whether a parameter is an MXU weight (matmul operand).

    MXU: weight matrices in linear/conv layers (q_proj, k_proj, v_proj,
         o_proj, gate_proj, up_proj, down_proj, fc1, fc2, patch_embedding,
         multi_modal_projector, projection layers).
    VPU: biases, normalization parameters (layernorm, rmsnorm, adaptive norm),
         embedding tables.
    """
    # Biases are always VPU
    if key.endswith(".bias"):
        return False
    # Normalization parameters are VPU (covers layernorm, rmsnorm, adaptive norm dense)
    if "norm" in key.lower():
        return False
    # Embedding lookups are VPU
    if "embed_tokens" in key or "position_embedding" in key:
        return False
    # Remaining .weight parameters are linear/conv weights -> MXU
    if key.endswith(".weight"):
        return True
    return False


# ---------------------------------------------------------------------------
# Quantization logic
# ---------------------------------------------------------------------------

def _quantize_tensor(
    tensor: torch.Tensor, target: QuantDtype
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Quantize a single tensor to the target dtype.

    For E4M3: per-tensor absmax scaling -> returns (quantized, scale).
    For FP16/BF16/FP32: simple cast -> returns (cast_tensor, None).
    """
    if target == QuantDtype.E4M3:
        tensor_f32 = tensor.float()
        amax = tensor_f32.abs().amax()
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        scale = (amax / fp8_max).clamp(min=1e-12).view(1)
        quantized = (tensor_f32 / scale).to(torch.float8_e4m3fn)
        return quantized, scale
    else:
        return tensor.to(QDTYPE_TO_TORCH[target]), None


def quantize_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    mxu_input: QuantDtype,
    mxu_output: QuantDtype,
    vpu_input: QuantDtype,
    vpu_output: QuantDtype,
) -> dict[str, torch.Tensor]:
    """Quantize a full state dict according to the MXU/VPU config.

    - MXU weight matrices are cast to ``mxu_input`` dtype.
      If E4M3, a companion ``{key}_scale`` tensor (in ``mxu_output`` dtype) is
      stored for dequantization: ``dequantized = quantized.to(mxu_output) * scale``.
    - VPU parameters (biases, norms, embeddings) are cast to ``vpu_input`` dtype.
      If E4M3, a companion ``{key}_scale`` tensor (in ``vpu_output`` dtype) is stored.
    """
    quantized: dict[str, torch.Tensor] = {}
    mxu_count = 0
    vpu_count = 0

    for key, tensor in state_dict.items():
        if is_mxu_weight(key):
            q, scale = _quantize_tensor(tensor, mxu_input)
            quantized[key] = q
            if scale is not None:
                quantized[f"{key}_scale"] = scale.to(QDTYPE_TO_TORCH[mxu_output])
            mxu_count += 1
        else:
            q, scale = _quantize_tensor(tensor, vpu_input)
            quantized[key] = q
            if scale is not None:
                quantized[f"{key}_scale"] = scale.to(QDTYPE_TO_TORCH[vpu_output])
            vpu_count += 1

    print(f"Quantized {mxu_count} MXU weight tensors to {mxu_input.value}, "
          f"{vpu_count} VPU tensors to {vpu_input.value}")
    return quantized


# ---------------------------------------------------------------------------
# Dequantization (for loading quantized safetensors at inference time)
# ---------------------------------------------------------------------------

def dequantize_state_dict(
    raw: dict[str, torch.Tensor],
    output_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Dequantize a state dict produced by ``quantize_state_dict``.

    For every key that has a companion ``{key}_scale`` tensor, reconstruct the
    full-precision weight:  ``dequantized = quantized.to(output_dtype) * scale``

    Non-quantized tensors are cast to ``output_dtype`` as-is.
    Scale tensors are consumed and not included in the output.

    Returns a state dict that is compatible with ``model.load_state_dict()``.
    """
    # Collect all scale keys first
    scale_keys = {k for k in raw if k.endswith("_scale")}
    base_keys_with_scale = {k.removesuffix("_scale") for k in scale_keys}

    dequantized: dict[str, torch.Tensor] = {}
    for key, tensor in raw.items():
        if key in scale_keys:
            continue  # consumed below
        if key in base_keys_with_scale:
            scale = raw[f"{key}_scale"]
            dequantized[key] = tensor.to(output_dtype) * scale.to(output_dtype)
        else:
            dequantized[key] = tensor.to(output_dtype)

    return dequantized


def load_quantized_model(
    model_config: "openpi.models.pi0_config.Pi0Config",
    weight_path: str,
    device: str = "cpu",
) -> "openpi.models_pytorch.pi0_pytorch.PI0Pytorch":
    """Load a quantized safetensors file into a PI0Pytorch model.

    1. Reads the raw quantized tensors (E4M3 + scales).
    2. Dequantizes everything back to bfloat16.
    3. Creates the model and loads the state dict.
    """
    raw = safetensors.torch.load_file(weight_path, device=device)
    state_dict = dequantize_state_dict(raw, output_dtype=torch.bfloat16)

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(config=model_config)
    model.load_state_dict(state_dict, strict=False)
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    return model


# ---------------------------------------------------------------------------
# Conversion entry point (mirrors convert_jax_model_to_pytorch but with quant)
# ---------------------------------------------------------------------------

def convert_pi0_checkpoint_quantized(
    checkpoint_dir: str,
    output_path: str,
    model_config: openpi.models.pi0_config.Pi0Config,
    *,
    mxu_input: QuantDtype,
    mxu_output: QuantDtype,
    vpu_input: QuantDtype,
    vpu_output: QuantDtype,
):
    """Convert PI0 JAX checkpoint to quantized PyTorch safetensors."""
    print(f"Converting PI0 checkpoint from {checkpoint_dir} to {output_path}")
    print(f"Quantization: MXU input={mxu_input.value}, MXU output={mxu_output.value}, "
          f"VPU input={vpu_input.value}, VPU output={vpu_output.value}")

    # Restore in float32 for accurate quantization
    initial_params = slice_initial_orbax_checkpoint(
        checkpoint_dir=checkpoint_dir, restore_precision="float32"
    )

    # --- projection params ---
    if model_config.pi05:
        keys = ["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"]
    else:
        keys = ["state_proj", "action_in_proj", "action_out_proj",
                "action_time_mlp_in", "action_time_mlp_out"]

    projection_params: dict[str, torch.Tensor] = {}
    for key in keys:
        kernel_params = initial_params["projection_params"][key]["kernel"]
        bias_params = initial_params["projection_params"][key]["bias"]
        if isinstance(kernel_params, dict):
            weight = kernel_params["value"]
            bias = bias_params["value"]
        else:
            weight = kernel_params
            bias = bias_params
        projection_params[f"{key}.weight"] = torch.from_numpy(np.array(weight)).T
        projection_params[f"{key}.bias"] = torch.from_numpy(np.array(bias))

    # --- PaliGemma config ---
    class PaliGemmaConfig:
        def __init__(self):
            self.vision_config = type("obj", (object,), {
                "hidden_size": 1152, "num_hidden_layers": 27,
                "num_attention_heads": 16, "intermediate_size": 4304,
                "patch_size": 14, "projection_dim": 2048,
            })()
            self.text_config = type("obj", (object,), {
                "hidden_size": 2048, "num_hidden_layers": 18,
                "num_attention_heads": 8, "head_dim": 256,
                "intermediate_size": 16384,
            })()

    paligemma_config = PaliGemmaConfig()
    action_expert_config = openpi.models.gemma.get_config("gemma_300m")

    # --- slice weights ---
    paligemma_params, expert_params = slice_paligemma_state_dict(
        initial_params["paligemma_params"], paligemma_config
    )
    gemma_params = slice_gemma_state_dict(
        expert_params, action_expert_config,
        num_expert=1, checkpoint_dir=checkpoint_dir, pi05=model_config.pi05,
    )

    # --- validate shapes via model ---
    pi0_model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_config)
    all_params = {**paligemma_params, **gemma_params, **projection_params}
    pi0_model.load_state_dict(all_params, strict=False)

    # Get the full state dict back (includes tied weights resolved)
    full_state_dict = pi0_model.float().state_dict()

    # --- quantize ---
    quantized_dict = quantize_state_dict(
        full_state_dict,
        mxu_input=mxu_input, mxu_output=mxu_output,
        vpu_input=vpu_input, vpu_output=vpu_output,
    )

    # --- save ---
    os.makedirs(output_path, exist_ok=True)
    safetensors.torch.save_file(quantized_dict, os.path.join(output_path, "model.safetensors"))

    # Copy assets folder if it exists
    assets_source = pathlib.Path(checkpoint_dir).parent / "assets"
    if assets_source.exists():
        assets_dest = pathlib.Path(output_path) / "assets"
        if assets_dest.exists():
            shutil.rmtree(assets_dest)
        shutil.copytree(assets_source, assets_dest)

    # Save config
    config_dict = {
        "action_dim": model_config.action_dim,
        "action_horizon": model_config.action_horizon,
        "paligemma_variant": model_config.paligemma_variant,
        "action_expert_variant": model_config.action_expert_variant,
        "quantization": {
            "mxu_input": mxu_input.value,
            "mxu_output": mxu_output.value,
            "vpu_input": vpu_input.value,
            "vpu_output": vpu_output.value,
        },
    }
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    print("Quantized model conversion completed successfully!")
    print(f"Model saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(
    checkpoint_dir: str,
    config_name: str,
    output_path: str | None = None,
    mxu_input: QuantDtype = QuantDtype.E4M3,
    mxu_output: QuantDtype = QuantDtype.BF16,
    vpu_input: QuantDtype = QuantDtype.BF16,
    vpu_output: QuantDtype = QuantDtype.BF16,
    *,
    inspect_only: bool = False,
):
    """Load JAX model and convert to quantized PyTorch safetensors.

    Args:
        checkpoint_dir: Path to the JAX checkpoint directory
        config_name: Training config name (e.g. pi0_droid, pi0_aloha_sim, pi05_droid)
        output_path: Path to save converted model (required unless --inspect_only)
        mxu_input: Dtype for MXU input weights (linear layer matrices)
        mxu_output: Dtype for MXU output (matmul accumulation)
        vpu_input: Dtype for VPU input params (biases, norms, embeddings)
        vpu_output: Dtype for VPU output
        inspect_only: Only inspect parameter keys, don't convert
    """
    # Enforce: MXU output must equal VPU input
    if mxu_output != vpu_input:
        raise ValueError(
            f"MXU output ({mxu_output.value}) must match VPU input ({vpu_input.value}). "
            f"Data flows from MXU output into VPU input, so they must use the same dtype."
        )

    model_config = _config.get_config(config_name).model
    if not isinstance(model_config, openpi.models.pi0_config.Pi0Config):
        raise ValueError(f"Config {config_name} is not a Pi0Config")

    if inspect_only:
        load_jax_model_and_print_keys(checkpoint_dir)
    else:
        if not output_path:
            print("Error: --output_path is required for conversion. Use --inspect_only to only view keys.")
            return
        convert_pi0_checkpoint_quantized(
            checkpoint_dir, output_path, model_config,
            mxu_input=mxu_input, mxu_output=mxu_output,
            vpu_input=vpu_input, vpu_output=vpu_output,
        )


if __name__ == "__main__":
    tyro.cli(main)
