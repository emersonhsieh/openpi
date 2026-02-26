"""Quantized SigLIP ViT — E4M3 MXU weights & activations, BF16 VPU.

Same architecture as siglip.py but every Dense / Conv matmul casts both
operands to float8_e4m3fn before the dot product, with BF16 accumulation.
LayerNorm and positional embeddings remain BF16 (VPU).

MXU input:  E4M3
MXU output: BF16
VPU input:  BF16
VPU output: BF16
"""

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import openpi.training.sharding as sharding

E4M3 = jnp.float8_e4m3fn
BF16 = jnp.bfloat16


# ---------------------------------------------------------------------------
# Positional embeddings (VPU — unchanged from siglip.py)
# ---------------------------------------------------------------------------

def posemb_sincos_2d(h, w, width, temperature=10_000.0, dtype=jnp.float32):
    y, x = jnp.mgrid[:h, :w]
    assert width % 4 == 0
    omega = jnp.arange(width // 4) / (width // 4 - 1)
    omega = 1.0 / (temperature**omega)
    y = jnp.einsum("m,d->md", y.flatten(), omega)
    x = jnp.einsum("m,d->md", x.flatten(), omega)
    pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
    return jnp.asarray(pe, dtype)[None, :, :]


def get_posemb(self, typ, seqshape, width, name, dtype=jnp.float32):
    if typ == "learn":
        return self.param(name, nn.initializers.normal(stddev=1 / np.sqrt(width)), (1, np.prod(seqshape), width), dtype)
    if typ == "sincos2d":
        return posemb_sincos_2d(*seqshape, width, dtype=dtype)
    raise ValueError(f"Unknown posemb type: {typ}")


# ---------------------------------------------------------------------------
# Quantised Dense — drop-in replacement for nn.Dense with E4M3 matmuls
# ---------------------------------------------------------------------------

class QuantizedDense(nn.Module):
    """Dense layer that casts weight + activation to E4M3 for the matmul."""

    features: int
    use_bias: bool = True
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x):
        kernel = self.param("kernel", self.kernel_init, (x.shape[-1], self.features))
        # MXU: E4M3 inputs → BF16 output
        y = jnp.dot(x.astype(E4M3), kernel.astype(E4M3), preferred_element_type=BF16)
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,))
            y = y + bias.astype(BF16)
        return y


# ---------------------------------------------------------------------------
# Quantised Conv — for patch embedding
# ---------------------------------------------------------------------------

class QuantizedConv(nn.Module):
    """Conv layer that casts weight + activation to E4M3 for the convolution."""

    features: int
    kernel_size: Sequence[int]
    strides: Sequence[int] | None = None
    padding: str = "SAME"

    @nn.compact
    def __call__(self, x):
        kernel_shape = (*self.kernel_size, x.shape[-1], self.features)
        kernel = self.param("kernel", nn.initializers.lecun_normal(), kernel_shape)
        bias = self.param("bias", nn.initializers.zeros_init(), (self.features,))
        strides = self.strides or (1,) * len(self.kernel_size)
        # MXU: E4M3 conv
        y = jax.lax.conv_general_dilated(
            x.astype(E4M3),
            kernel.astype(E4M3),
            window_strides=strides,
            padding=self.padding,
            preferred_element_type=BF16,
        )
        return y + bias.astype(BF16)


# ---------------------------------------------------------------------------
# Quantised MultiHeadDotProductAttention
# ---------------------------------------------------------------------------

class QuantizedMultiHeadDotProductAttention(nn.Module):
    """MHSA with E4M3 projections; softmax stays float32 (VPU)."""

    num_heads: int
    kernel_init: nn.initializers.Initializer = nn.initializers.xavier_uniform()
    deterministic: bool = True

    @nn.compact
    def __call__(self, inputs_q, inputs_kv):
        d = inputs_q.shape[-1]
        head_dim = d // self.num_heads

        # QKV projections — E4M3 MXU
        q = QuantizedDense(d, use_bias=False, kernel_init=self.kernel_init, name="query")(inputs_q)
        k = QuantizedDense(d, use_bias=False, kernel_init=self.kernel_init, name="key")(inputs_kv)
        v = QuantizedDense(d, use_bias=False, kernel_init=self.kernel_init, name="value")(inputs_kv)

        # Reshape to (B, L, num_heads, head_dim)
        B, Lq = q.shape[:2]
        Lk = k.shape[1]
        q = q.reshape(B, Lq, self.num_heads, head_dim)
        k = k.reshape(B, Lk, self.num_heads, head_dim)
        v = v.reshape(B, Lk, self.num_heads, head_dim)

        # Attention scores in float32 (VPU)
        scale = head_dim ** -0.5
        attn_weights = jnp.einsum("BQNH,BKNH->BNQK", q, k) * scale
        attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(BF16)

        # Weighted sum — E4M3 MXU
        attn_output = jnp.einsum(
            "BNQK,BKNH->BQNH",
            attn_weights.astype(E4M3),
            v.astype(E4M3),
            preferred_element_type=BF16,
        )
        attn_output = attn_output.reshape(B, Lq, d)

        # Output projection — E4M3 MXU
        return QuantizedDense(d, use_bias=False, kernel_init=self.kernel_init, name="out")(attn_output)


# ---------------------------------------------------------------------------
# Quantised MlpBlock
# ---------------------------------------------------------------------------

class MlpBlock(nn.Module):
    mlp_dim: int | None = None
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        inits = {
            "kernel_init": nn.initializers.xavier_uniform(),
            "bias_init": nn.initializers.normal(stddev=1e-6),
        }
        _, _, d = x.shape
        x = QuantizedDense(self.mlp_dim or 4 * d, **inits)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic)
        return QuantizedDense(d, **inits)(x)


# ---------------------------------------------------------------------------
# Quantised Encoder1DBlock
# ---------------------------------------------------------------------------

class Encoder1DBlock(nn.Module):
    mlp_dim: int | None = None
    num_heads: int = 12
    dropout: float = 0.0
    # dtype_mm kept for interface compat but MXU ops are always E4M3→BF16
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        out = {}
        x = sharding.activation_sharding_constraint(x)
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = out["sa"] = QuantizedMultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
        )(y, y)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+sa"] = x + y

        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = out["mlp"] = MlpBlock(mlp_dim=self.mlp_dim, dropout=self.dropout)(y, deterministic)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+mlp"] = x + y
        x = sharding.activation_sharding_constraint(x)
        return x, out


# ---------------------------------------------------------------------------
# Quantised Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    depth: int
    mlp_dim: int | None = None
    num_heads: int = 12
    dropout: float = 0.0
    scan: bool = False
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        out = {}
        if self.scan:
            block = nn.remat(
                Encoder1DBlock,
                prevent_cse=False,
                static_argnums=(2,),
                policy=getattr(jax.checkpoint_policies, self.remat_policy, None),
            )
            x, scan_out = nn.scan(
                block,
                variable_axes={"params": 0},
                split_rngs={"params": True, "dropout": True},
                in_axes=nn.broadcast,
                length=self.depth,
            )(
                name="encoderblock",
                dtype_mm=self.dtype_mm,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
            )(x, deterministic)
            for lyr in range(self.depth):
                out[f"block{lyr:02d}"] = jax.tree.map(lambda o, lyr=lyr: o[lyr], scan_out)
        else:
            for lyr in range(self.depth):
                block_cur = Encoder1DBlock(
                    name=f"encoderblock_{lyr}",
                    dtype_mm=self.dtype_mm,
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                )
                x, out[f"block{lyr:02d}"] = block_cur(x, deterministic)
            out["pre_ln"] = x

        return nn.LayerNorm(name="encoder_norm", dtype=self.dtype_mm)(x), out


# ---------------------------------------------------------------------------
# MAPHead (unchanged — only used when pool_type="map")
# ---------------------------------------------------------------------------

class MAPHead(nn.Module):
    mlp_dim: int | None = None
    num_heads: int = 12
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, x):
        n, _, d = x.shape
        probe = self.param("probe", nn.initializers.xavier_uniform(), (1, 1, d), x.dtype)
        probe = jnp.tile(probe, [n, 1, 1])
        x = QuantizedMultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
        )(probe, x)
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        x = x + MlpBlock(mlp_dim=self.mlp_dim)(y)
        return x[:, 0]


# ---------------------------------------------------------------------------
# Quantised _Module (ViT)
# ---------------------------------------------------------------------------

class _Module(nn.Module):
    """Quantised ViT model — E4M3 MXU, BF16 VPU."""

    num_classes: int | None = None
    patch_size: Sequence[int] = (16, 16)
    width: int = 768
    depth: int = 12
    mlp_dim: int | None = None
    num_heads: int = 12
    posemb: str = "learn"
    rep_size: int | bool = False
    dropout: float = 0.0
    pool_type: str = "gap"
    head_zeroinit: bool = True
    scan: bool = False
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, image, *, train=False):
        out = {}

        image = jnp.asarray(image, jnp.float32)

        # Patch extraction — quantised conv (E4M3 MXU)
        x = out["stem"] = QuantizedConv(
            self.width,
            self.patch_size,
            strides=self.patch_size,
            padding="VALID",
            name="embedding",
        )(image)

        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])

        # Positional embedding (VPU — float32 then cast)
        x = out["with_posemb"] = x + get_posemb(self, self.posemb, (h, w), c, "pos_embedding", jnp.float32)

        if self.pool_type == "tok":
            cls = self.param("cls", nn.initializers.zeros, (1, 1, c), x.dtype)
            x = jnp.concatenate([jnp.tile(cls, [n, 1, 1]), x], axis=1)

        n, _, c = x.shape
        x = nn.Dropout(rate=self.dropout)(x, not train)

        # Cast to BF16 for transformer blocks
        x = x.astype(BF16)

        x, out["encoder"] = Encoder(
            depth=self.depth,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            scan=self.scan,
            remat_policy=self.remat_policy,
            dtype_mm=self.dtype_mm,
            name="Transformer",
        )(x, deterministic=not train)
        encoded = out["encoded"] = x

        if self.pool_type == "map":
            x = out["head_input"] = MAPHead(num_heads=self.num_heads, mlp_dim=self.mlp_dim, dtype_mm=self.dtype_mm)(x)
        elif self.pool_type == "gap":
            x = out["head_input"] = jnp.mean(x, axis=1)
        elif self.pool_type == "0":
            x = out["head_input"] = x[:, 0]
        elif self.pool_type == "tok":
            x = out["head_input"] = x[:, 0]
            encoded = encoded[:, 1:]
        elif self.pool_type == "none":
            pass
        else:
            raise ValueError(f"Unknown pool type: '{self.pool_type}'")

        x_2d = jnp.reshape(encoded, [n, h, w, -1])

        if self.rep_size:
            rep_size = self.width if self.rep_size is True else self.rep_size
            hid = QuantizedDense(rep_size, name="pre_logits")
            x_2d = nn.tanh(hid(x_2d))
            x = nn.tanh(hid(x))

        out["pre_logits_2d"] = x_2d
        out["pre_logits"] = x

        if self.num_classes:
            kw = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}
            head = QuantizedDense(self.num_classes, name="head", **kw)
            x_2d = out["logits_2d"] = head(x_2d)
            x = out["logits"] = head(x)

        return x, out


def Module(num_classes=None, *, variant=None, **kw):  # noqa: N802
    """Factory function (same interface as siglip.Module)."""
    return _Module(num_classes, **{**decode_variant(variant), **kw})


def decode_variant(variant):
    """Converts a string like "B" or "B/32" into a params dict."""
    if variant is None:
        return {}
    v, patch = variant, {}
    if "/" in variant:
        v, patch = variant.split("/")
        patch = {"patch_size": (int(patch), int(patch))}
    return {
        "width": {"mu": 32, "Ti": 192, "S": 384, "M": 512, "B": 768, "L": 1024, "So400m": 1152, "H": 1280, "g": 1408, "g-opt": 1536, "G": 1664, "G-opt": 1536, "e": 1792}[v],
        "depth": {"mu": 1, "Ti": 12, "S": 12, "M": 12, "B": 12, "L": 24, "So400m": 27, "H": 32, "g": 40, "g-opt": 40, "G": 48, "G-opt": 48, "e": 56}[v],
        "mlp_dim": {"mu": 128, "Ti": 768, "S": 1536, "M": 2048, "B": 3072, "L": 4096, "So400m": 4304, "H": 5120, "g": 6144, "g-opt": 6144, "G": 8192, "G-opt": 8192, "e": 15360}[v],
        "num_heads": {"mu": 2, "Ti": 3, "S": 6, "M": 8, "B": 12, "L": 16, "So400m": 16, "H": 16, "g": 16, "g-opt": 16, "G": 16, "G-opt": 16, "e": 16}[v],
        **patch,
    }
