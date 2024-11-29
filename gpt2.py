import jax
import flax
from flax import nnx
import jax.numpy as jnp
from jax import random


class Attention(nnx.Module):
    def __init__(self,
                 d_in: int,
                 d_out: int,
                 context_length: int = 1024,
                 n_heads: int = 12,
                 dropout_rate: float = 0.1,
                 use_bias: bool = True,
                 rngs: nnx.Rngs = None):
        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"
        self.head_dim = d_out // n_heads
        self.d_int = d_in
        self.d_out = d_out
        self.context_length = context_length
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        #self.qkv_proj = nnx.Linear(d_in, d_out * 3, use_bias=use_bias)
        self.query_proj = nnx.Linear(d_in, d_out, use_bias=use_bias, rngs=rngs)
        self.key_proj = nnx.Linear(d_in, d_out, use_bias=use_bias, rngs=rngs)
        self.value_proj = nnx.Linear(d_in, d_out, use_bias=use_bias, rngs=rngs)
        self.attn_mask = jnp.tril(jnp.ones((context_length, context_length), dtype=jnp.float32))
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        self.out_proj = nnx.Linear(d_out, d_out, use_bias=use_bias, rngs=rngs)

    def __call__(self, x):
        b, l, d_in = x.shape
        queries = self.query_proj(x)    # (b, l, d_out)
        keys = self.key_proj(x)
        values = self.value_proj(x)
        # key, values = jnp.transpose(jnp.reshape(self.qkv(x), (b, l, 3, self.d_out)), (2, 0, 1, 3))
        # queries, keys, values = jnp.split(self.qkv_proj(x), 3, axis=-1)
        queries = jnp.transpose(jnp.reshape(queries, (b, l, self.n_heads, self.head_dim)),
                                (0, 2, 1, 3))
        keys = jnp.transpose(jnp.reshape(keys, (b, l, self.n_heads, self.head_dim)),
                             (0, 2, 1, 3))
        values = jnp.transpose(jnp.reshape(values, (b, l, self.n_heads, self.head_dim)),
                               (0, 2, 1, 3))
        attn_scores = jnp.matmul(queries, jnp.transpose(keys, (0, 1, 3, 2)))    # (b, n_heads, l, l)
        attn_mask_bool = self.attn_mask.astype(jnp.bool)[:l, :l]
        attn_scores = jnp.where(attn_mask_bool, attn_scores, -jnp.inf)
        attn_weights = jax.nn.softmax(attn_scores/jnp.sqrt(self.head_dim), axis=-1)
        attn_weights = self.dropout(attn_weights)
        contextualized_values = jnp.matmul(attn_weights, values)    # (b, n_heads, l, head_dim)
        contextualized_values = jnp.transpose(contextualized_values, (0, 2, 1, 3))  # (b, l, n_heads, head_dim)
        contextualized_values = jnp.reshape(contextualized_values, (b, l, self.d_out))
        contextualized_values = self.out_proj(contextualized_values)
        return contextualized_values


class FeedForward(nnx.Module):
    def __init__(self, emb_dim: int, rngs: nnx.Rngs = None):
        self.l1 = nnx.Linear(emb_dim, 4 * emb_dim, rngs=rngs)
        self.l2 = nnx.Linear(4 * emb_dim, emb_dim, rngs=rngs)

    def __call__(self, x):
        return self.l2(jax.nn.gelu(self.l1(x)))


class TransformerBlock(nnx.Module):
    def __init__(self,
                 emb_dim: int = 768,
                 context_length: int = 1024,
                 n_heads: int = 12,
                 dropout_rate: float = 0.1,
                 qkv_bias: bool = True,
                 rngs: nnx.Rngs = None):
        self.emb_dim = emb_dim
        self.context_length = context_length
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.qkv_bias = qkv_bias

        self.attn = Attention(
            d_in=emb_dim,
            d_out=emb_dim,
            context_length=context_length,
            n_heads=n_heads,
            dropout_rate=dropout_rate,
            use_bias=qkv_bias,
            rngs=rngs
        )
        self.ff = FeedForward(
            emb_dim=emb_dim,
            rngs=rngs
        )
        self.norm1 = nnx.LayerNorm(num_features=emb_dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(num_features=emb_dim, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, x):
        residue = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.dropout(x)
        x = x + residue
        residue = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + residue
        return x


class GPT2(nnx.Module):
    def __init__(self,
                 vocab_size: int = 50257,
                 emb_dim: int = 768,
                 context_length: int = 1024,
                 n_heads: int = 12,
                 n_layers: int = 12,
                 dropout_rate: float = 0.1,
                 qkv_bias: bool = True,
                 rngs: nnx.Rngs = None):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.context_length = context_length
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.qkv_bias = qkv_bias

        self.token_emb = nnx.Embed(num_embeddings=vocab_size, features=emb_dim, rngs=rngs)
        self.pos_emb = nnx.Embed(num_embeddings=context_length, features=emb_dim, rngs=rngs)
        self.drop_emb = nnx.Dropout(dropout_rate, rngs=rngs)
        self.transf_blocks = [TransformerBlock(emb_dim=emb_dim, context_length=context_length, n_heads=n_heads,
                                               dropout_rate=dropout_rate, qkv_bias=qkv_bias, rngs=rngs) for _ in range(n_layers)]
        self.last_norm = nnx.LayerNorm(num_features=emb_dim, rngs=rngs)
        self.out_linear = nnx.Linear(emb_dim, vocab_size, rngs=rngs, use_bias=False)

    def __call__(self, x):
        b, l = x.shape
        token_emb = self.token_emb(x)
        pos_emb = self.pos_emb(jnp.arange(l, device=x.device))
        x = token_emb + pos_emb
        x = self.drop_emb(x)
        for block in self.transf_blocks:
            x = block(x)
        x = self.last_norm(x)
        x = self.out_linear(x)
        return x
