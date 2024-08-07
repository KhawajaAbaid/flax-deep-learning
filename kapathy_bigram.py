import jax
from jax import random, numpy as jnp
import flax
from flax import linen as nn
from flax.training import train_state
import optax
from functools import partial
import os


batch_size = 32    # Reduce if you're gpu poor or use colab (like me)
block_size = 8

# ============================
#   DATA LOADING ETC
# ============================
filepath = './input.txt'
dl_path = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
if not os.path.exists(filepath):
    # download
    print("Downloading dataset...")
    os.system(f'wget {dl_path}')
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# mapping from chars to ints and vice versa
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = jnp.asarray(encode(text), dtype=jnp.int32)

# Train and test split
train_size = int(0.9 * len(data))
train_data = data[:train_size]
val_data = data[train_size:]
del data

def get_batch(split, key):
    data = train_data if split == 'train' else val_data
    ix = random.randint(key, (batch_size,), minval=0, maxval=len(data) - block_size)
    x = jnp.stack([data[i:i+block_size] for i in ix])
    y = jnp.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


# ==============================
# LETS NOW DEFINE THE MODEL
# =============================
class BigramLanguageModel(nn.Module):
    def setup(self):
        self.embedding_table = nn.Embed(num_embeddings=vocab_size, features=vocab_size)

    def __call__(self, idx):
        logits = self.embedding_table(idx)
        return logits


# ====================================
# PRE-TRAINING (GET IT? PRE-WORKOUT?)
# ====================================
slm = BigramLanguageModel()
key = random.PRNGKey(1999)
key, subkey = random.split(key)
params = slm.init(key, jnp.ones((1, block_size), dtype=jnp.int32))
state = train_state.TrainState.create(
    apply_fn=slm.apply,
    params=params['params'],
    tx=optax.adamw(learning_rate=3e-4),
)
eval_iters = 50


def estimate_loss(state, key):
    out = {}
    for split in ['train', 'val']:
        losses = jnp.zeros(eval_iters)
        key, subkey = random.split(key)
        for k in range(eval_iters):
            xb, yb = get_batch(split, key)
            logits = state.apply_fn({'params': state.params}, xb)
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, yb).mean()
            losses = losses.at[k].set(loss)
            key, subkey = random.split(subkey)
        out[split] = losses.mean()
    return out


# ============================
# TIME FOR TRAINING LES GOOO
# ============================
@jax.jit
def train_step(state, x, y):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, x)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        return loss
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state.apply_gradients(grads=grads)
    return loss, state


# Training loop
n_epochs = 1000
eval_interval = 50
key = random.PRNGKey(17771)
for epoch in range(n_epochs):

    key, subkey = random.split(key)

    if epoch % eval_interval == 0:
        losses = estimate_loss(state, key)
        key, subkey = random.split(subkey)
        print(
            f"Epoch {epoch}/{n_epochs} | "
            f"Train loss: {losses['train']:.4f} | "
            f"Val loss: {losses['val']:.4f}")

    xb, yb = get_batch('train', key)
    key, subkey = random.split(subkey)
    loss, state = train_step(state, xb, yb)
