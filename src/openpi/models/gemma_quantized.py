"""Quantized Gemma for Pi0 — E4M3 MXU weights & activations, BF16 VPU.

Same architecture as gemma.py but every matrix-multiply (MXU) operand is
cast to float8_e4m3fn before the dot product, with the accumulation /
output in bfloat16.  Vector-processing-unit (VPU) operations — RMSNorm,
biases, embeddings — remain in bfloat16.

MXU input:  E4M3   (both weight and activation operands)
MXU output: BF16   (matmul accumulation dtype)
VPU input:  BF16
VPU output: BF16
"""

from collections.abc import Sequence
import dataclasses
from typing import Literal, TypeAlias

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding

# Re-export constants from the original gemma module.
PALIGEMMA_VOCAB_SIZE = 257_152

E4M3 = jnp.float8_e4m3fn
BF16 = jnp.bfloat16


# ---------------------------------------------------------------------------
# Config — identical to gemma.Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Config:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    lora_configs: dict[str, lora.LoRAConfig] = dataclasses.field(default_factory=dict)


Variant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora"]


def get_config(variant: Variant) -> Config:
    """Returns config for specified gemma variant."""
    if variant == "dummy":
        return Config(width=64, depth=4, mlp_dim=128, num_heads=8, num_kv_heads=1, head_dim=16)
    if variant == "gemma_300m":
        return Config(width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256)
    if variant == "gemma_2b":
        return Config(width=2048, depth=18, mlp_dim=16_384, num_heads=8, num_kv_heads=1, head_dim=256)
    if variant == "gemma_2b_lora":
        return Config(
            width=2048, depth=18, mlp_dim=16_384, num_heads=8, num_kv_heads=1, head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=16, alpha=16.0), "ffn": lora.LoRAConfig(rank=16, alpha=16.0)},
        )
    if variant == "gemma_300m_lora":
        return Config(
            width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=32, alpha=32.0), "ffn": lora.LoRAConfig(rank=32, alpha=32.0)},
        )
    raise ValueError(f"Unknown variant: {variant}")


# ---------------------------------------------------------------------------
# Quantised matmul helpers
# ---------------------------------------------------------------------------

def _qdot(x, w, *, out_dtype=BF16):
    """Quantised dot: cast both operands to E4M3, accumulate in *out_dtype*."""
    return jnp.dot(
        x.astype(E4M3),
        w.astype(E4M3),
        preferred_element_type=out_dtype,
    )


def _qeinsum(eqn, x, w, *, out_dtype=BF16):
    """Quantised einsum: cast both operands to E4M3, accumulate in *out_dtype*."""
    return jnp.einsum(
        eqn,
        x.astype(E4M3),
        w.astype(E4M3),
        preferred_element_type=out_dtype,
    )


# ---------------------------------------------------------------------------
# Quantised Einsum (drop-in replacement for lora.Einsum)
# ---------------------------------------------------------------------------

class QuantizedEinsum(nn.Module):
    """Einsum with E4M3 quantised matmul and optional LoRA."""

    shape: tuple[int, ...]
    init_fn: nn.initializers.Initializer = nn.initializers.zeros
    lora_config: lora.LoRAConfig | None = None

    def setup(self):
        self.w = self.param("w", self.init_fn, self.shape)
        if config := self.lora_config:
            shape_a, shape_b = list(self.shape), list(self.shape)
            shape_a[config.axes[1]] = config.rank
            shape_b[config.axes[0]] = config.rank
            self.w_a = self.param("lora_a", config.init_fn, shape_a)
            self.w_b = self.param("lora_b", config.init_fn, shape_b)

    @nn.compact
    def __call__(self, eqn: str, x):
        # MXU: E4M3 inputs → BF16 output
        result = _qeinsum(eqn, x, self.w, out_dtype=BF16)

        if config := self.lora_config:
            eqn_a, eqn_b = lora.Einsum(
                shape=self.shape, init_fn=self.init_fn, lora_config=config
            )._make_lora_eqns(eqn)
            lo = _qeinsum(eqn_a, x, self.w_a, out_dtype=BF16)
            lo = _qeinsum(eqn_b, lo, self.w_b, out_dtype=BF16)
            result = result + lo * config.scaling_value

        return result


# ---------------------------------------------------------------------------
# Quantised FeedForward (drop-in replacement for lora.FeedForward)
# ---------------------------------------------------------------------------

class QuantizedFeedForward(nn.Module):
    """Feed-forward with E4M3 quantised matmul and optional LoRA."""

    features: int
    hidden_dim: int
    lora_config: lora.LoRAConfig | None = None

    def setup(self):
        self.w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        )
        self.w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        )
        self.w_gating_lora = None
        self.w_linear_lora = None
        if self.lora_config:
            self.w_gating_lora = (
                self.param("gating_einsum_lora_a", self.lora_config.init_fn, (2, self.features, self.lora_config.rank)),
                self.param("gating_einsum_lora_b", self.lora_config.init_fn, (2, self.lora_config.rank, self.hidden_dim)),
            )
            self.w_linear_lora = (
                self.param("linear_lora_a", self.lora_config.init_fn, (self.hidden_dim, self.lora_config.rank)),
                self.param("linear_lora_b", self.lora_config.init_fn, (self.lora_config.rank, self.features)),
            )

    @nn.compact
    def __call__(self, x):
        ff_gate = self._qdot(
            x, self.w_gating[0],
            None if self.w_gating_lora is None else (self.w_gating_lora[0][0], self.w_gating_lora[1][0]),
        )
        gate_value = nn.gelu(ff_gate)

        ff1 = self._qdot(
            x, self.w_gating[1],
            None if self.w_gating_lora is None else (self.w_gating_lora[0][1], self.w_gating_lora[1][1]),
        )
        activations = gate_value * ff1

        outputs = self._qdot(activations, self.w_linear, self.w_linear_lora)
        return outputs

    @staticmethod
    def _qdot(x, w, lora_weights):
        """E4M3 dot product with optional LoRA."""
        base = _qdot(x, w, out_dtype=BF16)
        if lora_weights is None:
            return base
        lo = _qdot(x, lora_weights[0], out_dtype=BF16)
        lo = _qdot(lo, lora_weights[1], out_dtype=BF16)
        return base + lo


# ---------------------------------------------------------------------------
# RMSNorm — VPU, stays BF16 (identical to gemma.RMSNorm)
# ---------------------------------------------------------------------------

@at.typecheck
class RMSNorm(nn.Module):
    @nn.compact
    def __call__(self, x, cond):
        dtype = x.dtype
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))
        if cond is None:
            scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
            normed_inputs = normed_inputs * (1 + scale)
            return normed_inputs.astype(dtype), None

        modulation = nn.Dense(x.shape[-1] * 3, kernel_init=nn.initializers.zeros, dtype=dtype)(cond)
        scale, shift, gate = jnp.split(modulation[:, None, :], 3, axis=-1)
        normed_inputs = normed_inputs * (1 + scale) + shift
        return normed_inputs.astype(dtype), gate


# ---------------------------------------------------------------------------
# Embedder — VPU, stays BF16 (identical to gemma.Embedder)
# ---------------------------------------------------------------------------

@at.typecheck
class Embedder(nn.Module):
    vocab_size: int
    embed_dim: int

    def setup(self):
        self.input_embedding_table = self.param(
            "input_embedding", nn.initializers.normal(), (self.vocab_size, self.embed_dim),
        )

    def encode(self, x):
        x = self.input_embedding_table[(x,)]
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
        return x

    def decode(self, x):
        return jnp.dot(x, self.input_embedding_table.T)


# ---------------------------------------------------------------------------
# Quantised Attention
# ---------------------------------------------------------------------------

@at.typecheck
class Attention(nn.Module):
    """Attention with E4M3 quantised QKV projections and output projection."""

    configs: Sequence[Config]

    @nn.compact
    def __call__(self, xs, positions, attn_mask, kv_cache):
        assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
        assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
        assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)

        qkvs = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                continue
            if config.num_kv_heads == config.num_heads:
                qkv_einsum = QuantizedEinsum(
                    shape=(3, config.num_heads, config.width, config.head_dim),
                    name=_name("qkv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                qkvs.append(qkv_einsum("3KDH,BSD->3BSKH", x))
            else:
                q_einsum = QuantizedEinsum(
                    shape=(config.num_heads, config.width, config.head_dim),
                    name=_name("q_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                    lora_config=config.lora_configs.get("attn"),
                )
                q = q_einsum("BTD,NDH->BTNH", x)
                kv_einsum = QuantizedEinsum(
                    shape=(2, config.num_kv_heads, config.width, config.head_dim),
                    name=_name("kv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                k, v = kv_einsum("2KDH,BSD->2BSKH", x)
                qkvs.append((q, k, v))

        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))

        q = _apply_rope(q, positions=positions)
        q *= self.configs[0].head_dim ** -0.5
        k = _apply_rope(k, positions=positions)

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = jnp.concatenate([cache_k, k], axis=1)
            v = jnp.concatenate([cache_v, v], axis=1)

        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)
        # Attention logits: keep in float32 for numerical stability (VPU-like)
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)

        if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
            raise ValueError(
                f"Attention mask with shape {attn_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}"
            )

        big_neg = -2.3819763e38
        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)

        probs = jax.nn.softmax(masked_logits, axis=-1).astype(BF16)

        # Attention weighted sum: E4M3 MXU
        encoded = jnp.einsum(
            "BKGTS,BSKH->BTKGH",
            probs.astype(E4M3),
            v.astype(E4M3),
            preferred_element_type=BF16,
        )
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        out = []
        start = 0
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                end = start + x.shape[1]
                out_einsum = QuantizedEinsum(
                    shape=(config.num_heads, config.head_dim, config.width),
                    name=_name("attn_vec_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                    lora_config=config.lora_configs.get("attn"),
                )
                out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
                start = end
            else:
                out.append(None)

        return out, (k, v)


# ---------------------------------------------------------------------------
# Quantised FeedForward (from gemma — not lora variant)
# ---------------------------------------------------------------------------

@at.typecheck
class FeedForwardSimple(nn.Module):
    """Non-LoRA feed-forward with E4M3 matmuls (mirrors gemma.FeedForward)."""

    features: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        )
        ff_gate = _qdot(x, w_gating[0], out_dtype=BF16)
        gate_value = nn.gelu(ff_gate)

        ff1 = _qdot(x, w_gating[1], out_dtype=BF16)
        activations = gate_value * ff1

        w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        )
        outputs = _qdot(activations, w_linear, out_dtype=BF16)
        return outputs


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------

@at.typecheck
class Block(nn.Module):
    """Transformer block — quantised matmuls, BF16 norms."""

    configs: tuple[Config, ...]
    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()

    @nn.compact
    def __call__(self, xs, kv_cache, positions, attn_mask, adarms_cond, deterministic=True):  # noqa: FBT002
        xs = sharding.activation_sharding_constraint(xs)
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x

        attn = Attention(configs=self.configs, name="attn")

        pre_attn = []
        gates = []
        for i, x in enumerate(xs):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_attention_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
            pre_attn.append(x)
            gates.append(gate if x is not None else None)

        pre_attn = sharding.activation_sharding_constraint(pre_attn)
        post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache)
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
        post_attn = sharding.activation_sharding_constraint(post_attn)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, post_attn, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        out = []
        gates = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_ffw_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
                x = QuantizedFeedForward(  # noqa: PLW2901
                    features=config.width,
                    hidden_dim=config.mlp_dim,
                    name=_name("mlp", i),
                    lora_config=config.lora_configs.get("ffn"),
                )(x)
            out.append(x)
            gates.append(gate if x is not None else None)

        out = sharding.activation_sharding_constraint(out)
        out = jax.tree.map(lambda x: drop(x, deterministic), out)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, out, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        return xs, kv_cache


# ---------------------------------------------------------------------------
# KVCache type alias
# ---------------------------------------------------------------------------

KVCache: TypeAlias = tuple[at.Float[at.Array, "l b _t _k _h"], at.Float[at.Array, "l b _t _v _h"]]


# ---------------------------------------------------------------------------
# Quantised Module (top-level Gemma)
# ---------------------------------------------------------------------------

@at.typecheck
class Module(nn.Module):
    """Quantised Gemma transformer — E4M3 MXU, BF16 VPU."""

    configs: Sequence[Config]
    embed_dtype: str

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()
    adarms: bool = False

    def setup(self):
        assert all(config.depth == self.configs[0].depth for config in self.configs)

        self.embedder = Embedder(
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            embed_dim=self.configs[0].width,
            name="embedder",
        )
        block_cls = nn.remat(
            Block,
            prevent_cse=False,
            static_argnums=(5,),
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(0, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast),
            length=self.configs[0].depth,
        )(
            configs=self.configs,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
        )
        self.final_norms = [RMSNorm(name=_name("final_norm", i)) for i in range(len(self.configs))]

    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    @at.typecheck
    def __call__(
        self,
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        positions: at.Int[at.Array, "b t"],
        mask: at.Bool[at.Array, "b t s"],
        adarms_cond: Sequence[at.Float[at.Array, "b _d"] | None] | None = None,
        *,
        kv_cache: KVCache | None = None,
        deterministic: bool = True,
    ) -> tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache]:
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        mask = jnp.asarray(mask)[:, None, :, :]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)

        embedded, kv_cache = self.layers(embedded, kv_cache, positions, mask, adarms_cond, deterministic)

        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)

        return [
            f(e, a)[0] if e is not None else e for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ], kv_cache

    def init(self, use_adarms: Sequence[bool]):
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
            adarms_cond=[jnp.zeros((1, c.width)) if u else None for u, c in zip(use_adarms, self.configs, strict=True)],
        )


# ---------------------------------------------------------------------------
# Helpers (identical to gemma.py)
# ---------------------------------------------------------------------------

def _apply_rope(x, *, positions, max_wavelength=10_000):
    freq_exponents = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2, dtype=jnp.float32)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]
    assert radians.dtype == jnp.float32
    sin, cos = jnp.sin(radians), jnp.cos(radians)
    x1, x2 = jnp.split(x, 2, axis=-1)
    res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    assert res.dtype == jnp.float32
    return res.astype(x.dtype)


def _name(name, i):
    if i == 0:
        return name
    return f"{name}_{i}"


def _gated_residual(x, y, gate):
    assert (x is None) == (y is None)
    if x is None:
        return None
    if gate is None:
        return x + y
    return x + y * gate
